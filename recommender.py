"""
recommender.py
Xử lý dữ liệu và xây dựng mô hình KNN (Nearest Neighbors) để gợi ý xe
dựa trên tiêu chí khách hàng nhập vào.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

NUMERIC_COLS = ["Giá( VNĐ)", "Công suất(HP)", "Chỗ ngồi"]
CATEGORICAL_COLS = ["Nhiên liệu", "Mục đích", "Tình trạng", "Kiểu xe", "Hộp số"]

# Trọng số mặc định cho từng thuộc tính khi tính khoảng cách.
# Giá và Công suất được ưu tiên cao hơn theo yêu cầu của đồ án.
DEFAULT_WEIGHTS = {
    "Giá( VNĐ)": 2.0,
    "Công suất(HP)": 1.5,
    "Chỗ ngồi": 1.0,
    "Nhiên liệu": 1.0,
    "Mục đích": 1.0,
    "Tình trạng": 0.5,
    "Kiểu xe": 1.0,
    "Hộp số": 0.5,
}


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Đọc CSV gốc và làm sạch dữ liệu."""
    df = pd.read_csv(path)

    # Giá: "458.000.000" -> 458000000
    df["Giá( VNĐ)"] = (
        df["Giá( VNĐ)"].astype(str).str.replace(".", "", regex=False).astype(float)
    )

    # Công suất: chuỗi số (có thể dùng dấu phẩy thập phân kiểu VN, vd "134,1") -> float
    df["Công suất(HP)"] = (
        df["Công suất(HP)"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )

    # Chuẩn hóa lỗi chính tả Nhiên liệu: "HYBRID" / "Hybrid" -> "Hybrid"
    df["Nhiên liệu"] = df["Nhiên liệu"].str.strip().str.title()
    df["Nhiên liệu"] = df["Nhiên liệu"].replace({"Hybrid": "Hybrid"})

    # Chuẩn hóa khoảng trắng thừa ở các cột phân loại khác
    for col in CATEGORICAL_COLS:
        if col != "Nhiên liệu":
            df[col] = df[col].astype(str).str.strip()

    df = df.reset_index(drop=True)
    return df


class CarRecommender:
    """
    Bọc toàn bộ pipeline: chuẩn hóa số, one-hot encoding, áp trọng số,
    và tìm K hàng xóm gần nhất bằng NearestNeighbors.
    """

    def __init__(self, df: pd.DataFrame, weights: dict | None = None):
        self.df = df.reset_index(drop=True)
        self.weights = weights or DEFAULT_WEIGHTS

        # --- Chuẩn hóa các cột số về [0, 1] ---
        self.scaler = MinMaxScaler()
        self.numeric_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df[NUMERIC_COLS]),
            columns=NUMERIC_COLS,
        )

        # --- One-hot encoding cho các cột phân loại ---
        self.encoded_cat = pd.get_dummies(self.df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS)

        # Ghi nhớ để tra cứu cột nào thuộc thuộc tính gốc nào (dùng khi áp trọng số)
        self.cat_col_map = {}
        for col in CATEGORICAL_COLS:
            self.cat_col_map[col] = [c for c in self.encoded_cat.columns if c.startswith(f"{col}_")]

        self.feature_matrix = self._build_weighted_matrix()

        self.model = NearestNeighbors(metric="euclidean")
        self.model.fit(self.feature_matrix.values)

    def _build_weighted_matrix(self) -> pd.DataFrame:
        parts = []
        for col in NUMERIC_COLS:
            w = self.weights.get(col, 1.0)
            parts.append(self.numeric_scaled[[col]] * w)
        for col in CATEGORICAL_COLS:
            w = self.weights.get(col, 1.0)
            sub_cols = self.cat_col_map[col]
            if sub_cols:
                parts.append(self.encoded_cat[sub_cols] * w)
        return pd.concat(parts, axis=1)

    def _encode_query(self, query: dict) -> np.ndarray:
        """
        Chuyển 1 tiêu chí khách hàng (dict) thành vector cùng không gian
        đặc trưng với feature_matrix. Thuộc tính không nhập sẽ được gán
        giá trị trung tính (không lệch về hướng nào).
        """
        row = {}

        # --- Số: nếu khách không nhập, dùng giá trị trung bình (đã scale = ~0.5 vùng giữa) ---
        numeric_input = {
            "Giá( VNĐ)": query.get("Giá( VNĐ)"),
            "Công suất(HP)": query.get("Công suất(HP)"),
            "Chỗ ngồi": query.get("Chỗ ngồi"),
        }
        numeric_df = pd.DataFrame([numeric_input])
        for col in NUMERIC_COLS:
            if pd.isna(numeric_df[col].iloc[0]):
                numeric_df[col] = self.numeric_scaled[col].mean() / (self.weights.get(col, 1.0) or 1)
        scaled = self.scaler.transform(numeric_df[NUMERIC_COLS])[0]
        for i, col in enumerate(NUMERIC_COLS):
            row[col] = scaled[i] * self.weights.get(col, 1.0)

        # --- Phân loại: one-hot, nếu không chọn thì để toàn bộ = 0 (không thiên vị) ---
        for col in CATEGORICAL_COLS:
            sub_cols = self.cat_col_map[col]
            chosen = query.get(col)
            for sc in sub_cols:
                value = 1.0 if (chosen is not None and sc == f"{col}_{chosen}") else 0.0
                row[sc] = value * self.weights.get(col, 1.0)

        vector = pd.DataFrame([row])[self.feature_matrix.columns].values[0]
        return vector

    def recommend(self, query: dict, k: int = 5) -> pd.DataFrame:
        """Trả về top-k xe gần nhất với tiêu chí query."""
        vector = self._encode_query(query).reshape(1, -1)
        k = min(k, len(self.df))
        distances, indices = self.model.kneighbors(vector, n_neighbors=k)

        # Chuẩn hóa "độ phù hợp" theo khoảng cách xa nhất trong TOÀN BỘ dataset
        # (không chỉ trong top-k) để phần trăm phản ánh đúng mức tương đồng.
        all_distances, _ = self.model.kneighbors(vector, n_neighbors=len(self.df))
        max_d = all_distances.max() or 1.0

        result = self.df.iloc[indices[0]].copy()
        result["Khoảng cách"] = distances[0]
        result["Độ phù hợp (%)"] = ((1 - result["Khoảng cách"] / (max_d + 1e-9)) * 100).round(1)
        result = result.sort_values("Khoảng cách").reset_index(drop=True)
        return result

    def unique_values(self, col: str):
        return sorted(self.df[col].dropna().unique().tolist())