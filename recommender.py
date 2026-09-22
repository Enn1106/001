"""
recommender.py

Xử lý dữ liệu và xây dựng mô hình KNN (Nearest Neighbors)
để gợi ý xe dựa trên tiêu chí khách hàng nhập vào.

Luồng chính:
- Làm sạch dữ liệu CSV
- Log-transform Giá và Công suất để giảm ảnh hưởng của giá trị ngoại lệ
- Chuẩn hóa thuộc tính số bằng MinMaxScaler về [0, 1]
- One-Hot Encoding thuộc tính phân loại
- Áp dụng trọng số cho từng thuộc tính
- Dùng KNN + Euclidean Distance để tìm Top-K xe gần nhất
- Tính mức độ đáp ứng từng tiêu chí để giải thích kết quả
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors


NUMERIC_COLS = [
    "Giá( VNĐ)",
    "Công suất(HP)",
    "Chỗ ngồi",
]

# Không đưa "Giá" và "Công suất" trực tiếp vào MinMaxScaler.
# Hai thuộc tính này sẽ được log1p trước rồi mới MinMaxScaler.
LOG_NUMERIC_COLS = [
    "Giá( VNĐ)",
    "Công suất(HP)",
]

CATEGORICAL_COLS = [
    "Nhiên liệu",
    "Mục đích",
    "Tình trạng",
    "Kiểu xe",
    "Hộp số",
]


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


def load_and_clean_data(file_path):
    """
    Đọc và làm sạch dữ liệu từ CSV.
    """

    df = pd.read_csv(file_path)

    # -----------------------------
    # Làm sạch dữ liệu số
    # -----------------------------
    if "Giá( VNĐ)" in df.columns:
        df["Giá( VNĐ)"] = (
            df["Giá( VNĐ)"]
            .astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        df["Giá( VNĐ)"] = pd.to_numeric(
            df["Giá( VNĐ)"],
            errors="coerce"
        )

    if "Công suất(HP)" in df.columns:
        df["Công suất(HP)"] = (
            df["Công suất(HP)"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df["Công suất(HP)"] = pd.to_numeric(
            df["Công suất(HP)"],
            errors="coerce"
        )

    if "Chỗ ngồi" in df.columns:
        df["Chỗ ngồi"] = pd.to_numeric(
            df["Chỗ ngồi"],
            errors="coerce"
        )

    # -----------------------------
    # Làm sạch dữ liệu phân loại
    # -----------------------------
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Chuẩn hóa riêng Nhiên liệu:
    # HYBRID / Hybrid / hybrid -> Hybrid
    if "Nhiên liệu" in df.columns:
        df["Nhiên liệu"] = df["Nhiên liệu"].str.title()

    # -----------------------------
    # Xóa dòng thiếu dữ liệu cần thiết
    # -----------------------------
    required_cols = NUMERIC_COLS + CATEGORICAL_COLS
    required_cols = [
        col for col in required_cols
        if col in df.columns
    ]

    df = df.dropna(subset=required_cols)

    # Xóa dòng trùng hoàn toàn
    df = df.drop_duplicates().reset_index(drop=True)

    return df


class CarRecommender:
    def __init__(self, df, weights=None):
        self.df = df.copy()

        self.weights = (
            DEFAULT_WEIGHTS.copy()
            if weights is None
            else weights.copy()
        )

        # -----------------------------
        # Chuẩn hóa dữ liệu số
        # -----------------------------
        self.scaler = MinMaxScaler()

        numeric_data = self.df[NUMERIC_COLS].copy()

        # Log-transform Giá và Công suất.
        # Chỗ ngồi giữ nguyên vì đây là thuộc tính số có phạm vi nhỏ.
        for col in LOG_NUMERIC_COLS:
            numeric_data[col] = np.log1p(numeric_data[col])

        self.scaled_numeric = pd.DataFrame(
            self.scaler.fit_transform(numeric_data),
            columns=NUMERIC_COLS,
            index=self.df.index
        )

        # -----------------------------
        # One-Hot Encoding
        # -----------------------------
        self.encoded_categorical = pd.get_dummies(
            self.df[CATEGORICAL_COLS],
            dtype=float
        )

        # Lưu mapping để encode query giống dữ liệu training
        self.cat_col_map = {}

        for col in CATEGORICAL_COLS:
            prefix = f"{col}_"

            self.cat_col_map[col] = [
                encoded_col
                for encoded_col in self.encoded_categorical.columns
                if encoded_col.startswith(prefix)
            ]

    def _active_columns(self, query):
        """
        Chỉ lấy các tiêu chí thực sự được người dùng yêu cầu
        và có trọng số > 0.

        Điều này giúp:
        "Không yêu cầu" -> không tham gia vào khoảng cách KNN.
        """

        active_columns = []

        for col in NUMERIC_COLS + CATEGORICAL_COLS:
            value = query.get(col)

            if value is None:
                continue

            if isinstance(value, str):
                if value.strip() == "":
                    continue

                if value.strip().lower() == "không yêu cầu":
                    continue

            weight = self.weights.get(col, 1.0)

            if weight <= 0:
                continue

            active_columns.append(col)

        return active_columns

    def _transform_numeric_query(self, col, value):
        """
        Biến đổi query số theo đúng pipeline:
        log1p -> MinMaxScaler.
        """

        value = float(value)

        if col in LOG_NUMERIC_COLS:
            value = np.log1p(value)

        # scaler.transform cần đủ số cột theo thứ tự ban đầu.
        raw_values = self.df[NUMERIC_COLS].copy()

        for numeric_col in LOG_NUMERIC_COLS:
            raw_values[numeric_col] = np.log1p(
                raw_values[numeric_col]
            )

        temp_scaler = MinMaxScaler()
        temp_scaler.fit(raw_values)

        query_array = raw_values.iloc[0:1].copy()
        query_array[col] = value

        scaled = temp_scaler.transform(query_array)

        col_index = NUMERIC_COLS.index(col)

        return float(scaled[0, col_index])

    def _build_weighted_matrix(self, active_columns):
        """
        Tạo ma trận đặc trưng sau khi áp dụng trọng số.

        Thứ tự:
        dữ liệu đã chuẩn hóa/encode
        -> chọn tiêu chí active
        -> nhân trọng số
        """

        feature_parts = []

        for col in active_columns:

            if col in NUMERIC_COLS:
                values = self.scaled_numeric[[col]].to_numpy()

                weight = self.weights.get(col, 1.0)

                values = values * weight

                feature_parts.append(values)

            else:
                encoded_cols = self.cat_col_map[col]

                values = (
                    self.encoded_categorical[encoded_cols]
                    .to_numpy()
                )

                weight = self.weights.get(col, 1.0)

                values = values * weight

                feature_parts.append(values)

        if not feature_parts:
            return np.empty((len(self.df), 0))

        return np.hstack(feature_parts)

    def _encode_query(self, query, active_columns):
        """
        Encode query vào đúng không gian đặc trưng với dữ liệu xe.
        """

        query_features = []

        for col in active_columns:

            if col in NUMERIC_COLS:

                value = query[col]

                # Tạo vector đúng thứ tự để dùng scaler đã fit.
                raw_query = self.df[NUMERIC_COLS].iloc[0:1].copy()

                raw_query[col] = float(value)

                for log_col in LOG_NUMERIC_COLS:
                    raw_query[log_col] = np.log1p(
                        raw_query[log_col]
                    )

                scaled = self.scaler.transform(raw_query)

                col_index = NUMERIC_COLS.index(col)

                query_value = scaled[0, col_index]

                weight = self.weights.get(col, 1.0)

                query_features.append([
                    float(query_value) * weight
                ])

            else:

                encoded_cols = self.cat_col_map[col]

                query_vector = np.zeros(
                    len(encoded_cols),
                    dtype=float
                )

                query_value = str(query[col]).strip()

                target_col = f"{col}_{query_value}"

                if target_col in encoded_cols:
                    index = encoded_cols.index(target_col)
                    query_vector[index] = 1.0

                weight = self.weights.get(col, 1.0)

                query_features.append(
                    query_vector * weight
                )

        if not query_features:
            return np.empty((1, 0))

        return np.concatenate(query_features).reshape(1, -1)

    def _criterion_scores(self, car, query):
        """
        Tính điểm đáp ứng từng tiêu chí để giải thích kết quả.

        Điểm này KHÔNG phải xác suất.

        Với thuộc tính số:
            score = max(0, 1 - |actual-target| / target) * 100

        Với thuộc tính phân loại:
            khớp = 100
            không khớp = 0
        """

        scores = {}

        for col in NUMERIC_COLS:

            target = query.get(col)

            if target is None:
                continue

            if isinstance(target, str):
                if target.strip().lower() == "không yêu cầu":
                    continue

            try:
                target = float(target)
                actual = float(car[col])

                if target == 0:
                    score = (
                        100.0
                        if actual == 0
                        else 0.0
                    )
                else:
                    score = max(
                        0.0,
                        1.0 - abs(actual - target) / abs(target)
                    ) * 100.0

                scores[col] = round(score, 1)

            except (TypeError, ValueError):
                continue

        for col in CATEGORICAL_COLS:

            target = query.get(col)

            if target is None:
                continue

            if str(target).strip().lower() == "không yêu cầu":
                continue

            actual = str(car[col]).strip()
            target = str(target).strip()

            scores[col] = (
                100.0
                if actual.lower() == target.lower()
                else 0.0
            )

        return scores

    def _overall_criterion_score(self, scores):
        """
        Tính điểm đáp ứng tổng thể bằng trung bình có trọng số.
        """

        if not scores:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for col, score in scores.items():
            weight = self.weights.get(col, 1.0)

            if weight <= 0:
                continue

            total_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return round(
            total_score / total_weight,
            1
        )

    def _build_explanations(self, scores):
        """
        Sinh giải thích đơn giản, dễ trình bày trong đồ án.
        """

        explanations = []

        for col, score in scores.items():

            # -------------------------
            # Giá
            # -------------------------
            if col == "Giá( VNĐ)":

                if score >= 95:
                    text = "✓ Giá rất gần ngân sách"

                elif score >= 80:
                    text = "✓ Giá tương đối gần ngân sách"

                else:
                    text = (
                        "△ Giá chênh lệch khá nhiều "
                        "so với ngân sách"
                    )

            # -------------------------
            # Công suất
            # -------------------------
            elif col == "Công suất(HP)":

                if score >= 95:
                    text = "✓ Công suất gần mức mong muốn"

                elif score >= 80:
                    text = "△ Công suất có chênh lệch nhỏ"

                else:
                    text = (
                        "△ Công suất chênh lệch đáng kể "
                        "so với nhu cầu"
                    )

            # -------------------------
            # Chỗ ngồi
            # -------------------------
            elif col == "Chỗ ngồi":

                if score == 100:
                    text = "✓ Đúng số chỗ yêu cầu"

                elif score >= 80:
                    text = "△ Số chỗ gần với nhu cầu"

                else:
                    text = "△ Số chỗ chênh lệch so với nhu cầu"

            # -------------------------
            # Thuộc tính phân loại
            # -------------------------
            else:

                labels = {
                    "Nhiên liệu": "nhiên liệu",
                    "Mục đích": "mục đích sử dụng",
                    "Tình trạng": "tình trạng",
                    "Kiểu xe": "kiểu xe",
                    "Hộp số": "hộp số",
                }

                label = labels.get(col, col)

                if score == 100:
                    text = f"✓ Đúng {label} yêu cầu"
                else:
                    text = f"△ Không khớp {label} yêu cầu"

            explanations.append(text)

        return explanations

    def recommend(self, query, k=5):
        """
        Trả về Top-K xe gần nhu cầu nhất bằng KNN.

        KNN sử dụng Euclidean Distance.
        """

        active_columns = self._active_columns(query)

        if not active_columns:
            raise ValueError(
                "Bạn cần nhập ít nhất một tiêu chí tìm kiếm."
            )

        X = self._build_weighted_matrix(
            active_columns
        )

        query_vector = self._encode_query(
            query,
            active_columns
        )

        # -----------------------------
        # KNN
        # -----------------------------
        n_neighbors = min(
            k,
            len(self.df)
        )

        model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="euclidean"
        )

        model.fit(X)

        distances, indices = model.kneighbors(
            query_vector
        )

        result = self.df.iloc[
            indices[0]
        ].copy()

        result["Khoảng cách KNN"] = np.round(
            distances[0],
            4
        )

        overall_scores = []
        details = []
        explanations = []

        for idx in result.index:

            car = self.df.loc[idx]

            scores = self._criterion_scores(
                car,
                query
            )

            overall_score = (
                self._overall_criterion_score(
                    scores
                )
            )

            overall_scores.append(
                overall_score
            )

            details.append(scores)

            explanations.append(
                self._build_explanations(
                    scores
                )
            )

        result["Điểm đáp ứng (%)"] = (
            overall_scores
        )

        result["Chi tiết tiêu chí"] = details

        result["Giải thích"] = explanations

        # Sắp xếp theo đúng thứ tự KNN
        result = result.reset_index(drop=True)

        return result

    def unique_values(self, column):
        """
        Lấy danh sách giá trị duy nhất của một cột.
        """

        if column not in self.df.columns:
            return []

        return sorted(
            self.df[column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
