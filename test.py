"""
test.py
Hệ thống hỗ trợ lựa chọn xe ô tô theo nhu cầu, dùng thuật toán KNN.

Chạy bằng:
    streamlit run test.py
"""

import os
import pandas as pd
import streamlit as st

from recommender import (
    DEFAULT_WEIGHTS,
    CarRecommender,
    load_and_clean_data,
)


st.set_page_config(
    page_title="Gợi ý xe ô tô theo nhu cầu",
    layout="wide",
)


@st.cache_data
def get_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "cars.csv")
    return load_and_clean_data(csv_path)


@st.cache_resource
def get_recommender(_df: pd.DataFrame, weights: dict):
    return CarRecommender(_df, weights=weights)


def format_vnd(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".") + " VNĐ"


def show_score_bar(label: str, score: float):
    # Phân loại màu sắc và độ nổi bật dựa trên điểm số
    if score >= 80:
        color = "#2e7d32"  # Xanh lá cây (Phù hợp)
        text_style = f"color: {color}; font-weight: bold;"
    elif score >= 50:
        color = "#f57c00"  # Cam (Tạm ổn)
        text_style = "color: #555; font-weight: normal;"
    else:
        color = "#d32f2f"  # Đỏ (Không phù hợp / Điểm thấp)
        text_style = f"color: {color}; font-weight: normal;"

    html_str = f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="{text_style}">{label}</span>
            <span style="{text_style}">{score:.1f}%</span>
        </div>
        <div style="width: 100%; background-color: #e0e0e0; border-radius: 4px; height: 8px;">
            <div style="width: {score}%; background-color: {color}; height: 100%; border-radius: 4px;"></div>
        </div>
    </div>
    """
    st.markdown(html_str, unsafe_allow_html=True)


df = get_data()

st.title("Hệ thống hỗ trợ lựa chọn xe ô tô theo nhu cầu sử dụng")
st.caption(
    "Đồ án môn Hệ trợ giúp quyết định — Gợi ý xe bằng thuật toán "
    "K-Nearest Neighbors (KNN)"
)

with st.sidebar:
    st.header("Trọng số thuộc tính")
    st.caption(
        "Trọng số thể hiện mức độ quan trọng của từng tiêu chí "
        "khi tính khoảng cách KNN."
    )

    w_price = st.slider("Giá", 0.0, 3.0, DEFAULT_WEIGHTS["Giá( VNĐ)"], 0.1)
    w_power = st.slider("Công suất", 0.0, 3.0, DEFAULT_WEIGHTS["Công suất(HP)"], 0.1)
    w_seats = st.slider("Chỗ ngồi", 0.0, 3.0, DEFAULT_WEIGHTS["Chỗ ngồi"], 0.1)
    w_fuel = st.slider("Nhiên liệu", 0.0, 3.0, DEFAULT_WEIGHTS["Nhiên liệu"], 0.1)
    w_purpose = st.slider("Mục đích", 0.0, 3.0, DEFAULT_WEIGHTS["Mục đích"], 0.1)
    w_body = st.slider("Kiểu xe", 0.0, 3.0, DEFAULT_WEIGHTS["Kiểu xe"], 0.1)
    w_condition = st.slider("Tình trạng", 0.0, 3.0, DEFAULT_WEIGHTS["Tình trạng"], 0.1)
    w_gearbox = st.slider("Hộp số", 0.0, 3.0, DEFAULT_WEIGHTS["Hộp số"], 0.1)

    weights = {
        "Giá( VNĐ)": w_price,
        "Công suất(HP)": w_power,
        "Chỗ ngồi": w_seats,
        "Nhiên liệu": w_fuel,
        "Mục đích": w_purpose,
        "Kiểu xe": w_body,
        "Tình trạng": w_condition,
        "Hộp số": w_gearbox,
    }

recommender = get_recommender(df, weights)

st.subheader("1. Nhập tiêu chí mong muốn")
st.caption(
    "Bỏ chọn hoặc chọn 'Không yêu cầu' nếu tiêu chí đó không quan trọng. "
    "Tiêu chí không yêu cầu sẽ không tham gia khoảng cách KNN."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    use_price = st.checkbox("Có yêu cầu về giá", value=True)
    price = st.slider(
        "Ngân sách (VNĐ)",
        min_value=int(df["Giá( VNĐ)"].min()),
        max_value=int(df["Giá( VNĐ)"].max()),
        value=int(df["Giá( VNĐ)"].median()),
        step=10_000_000,
        disabled=not use_price,
        format="%d",
    )

with col2:
    use_power = st.checkbox("Có yêu cầu về công suất", value=False)
    power = st.slider(
        "Công suất mong muốn (HP)",
        min_value=int(df["Công suất(HP)"].min()),
        max_value=int(df["Công suất(HP)"].max()),
        value=int(df["Công suất(HP)"].median()),
        disabled=not use_power,
    )

with col3:
    seats_options = ["Không yêu cầu"] + recommender.unique_values("Chỗ ngồi")
    seats = st.selectbox("Số chỗ ngồi", seats_options)

    fuel_options = ["Không yêu cầu"] + recommender.unique_values("Nhiên liệu")
    fuel = st.selectbox("Nhiên liệu", fuel_options)

with col4:
    purpose_options = ["Không yêu cầu"] + recommender.unique_values("Mục đích")
    purpose = st.selectbox("Mục đích sử dụng", purpose_options)

    body_options = ["Không yêu cầu"] + recommender.unique_values("Kiểu xe")
    body = st.selectbox("Kiểu xe", body_options)

col5, col6, col7 = st.columns(3)

with col5:
    condition_options = ["Không yêu cầu"] + recommender.unique_values("Tình trạng")
    condition = st.selectbox("Tình trạng", condition_options)

with col6:
    gearbox_options = ["Không yêu cầu"] + recommender.unique_values("Hộp số")
    gearbox = st.selectbox("Hộp số", gearbox_options)

with col7:
    k = st.number_input("Số lượng xe gợi ý (K)", min_value=1, max_value=20, value=5)

st.divider()

if st.button("Tìm xe phù hợp", type="primary", use_container_width=False):
    query = {}

    if use_price:
        query["Giá( VNĐ)"] = price
    if use_power:
        query["Công suất(HP)"] = power
    if seats != "Không yêu cầu":
        query["Chỗ ngồi"] = int(seats)
    if fuel != "Không yêu cầu":
        query["Nhiên liệu"] = fuel
    if purpose != "Không yêu cầu":
        query["Mục đích"] = purpose
    if body != "Không yêu cầu":
        query["Kiểu xe"] = body
    if condition != "Không yêu cầu":
        query["Tình trạng"] = condition
    if gearbox != "Không yêu cầu":
        query["Hộp số"] = gearbox

    active_query = {
        col: value
        for col, value in query.items()
        if weights.get(col, 1.0) > 0
    }

    if not active_query:
        st.warning(
            "Vui lòng nhập ít nhất một tiêu chí và đặt trọng số của "
            "tiêu chí đó lớn hơn 0."
        )
    else:
        result = recommender.recommend(active_query, k=int(k))

        # Sắp xếp bảng kết quả theo điểm đáp ứng cao nhất giảm dần
        result = result.sort_values(by="Điểm đáp ứng (%)", ascending=False).reset_index(drop=True)

        st.subheader("2. Kết quả gợi ý")
        st.caption(
            f"Top {len(result)} xe được xếp hạng bằng khoảng cách Euclidean có trọng số của KNN."
        )

        display_df = result[
            [
                "Tên xe",
                "Giá( VNĐ)",
                "Nhiên liệu",
                "Công suất(HP)",
                "Chỗ ngồi",
                "Mục đích",
                "Tình trạng",
                "Kiểu xe",
                "Hộp số",
                "Điểm đáp ứng (%)",
            ]
        ].copy()

        display_df["Giá( VNĐ)"] = display_df["Giá( VNĐ)"].apply(format_vnd)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("3. Giải thích kết quả")

        for index, row in result.iterrows():
            with st.expander(
                f"{index + 1}. {row['Tên xe']} — "
                f"Điểm đáp ứng {row['Điểm đáp ứng (%)']:.1f}%"
            ):
                col_info, col_score = st.columns([2, 1])

                with col_info:
                    st.write(f"**Giá:** {format_vnd(row['Giá( VNĐ)'])}")
                    st.write(f"**Nhiên liệu:** {row['Nhiên liệu']}")
                    st.write(f"**Công suất:** {row['Công suất(HP)']} HP")
                    st.write(f"**Chỗ ngồi:** {row['Chỗ ngồi']}")
                    st.write(f"**Mục đích:** {row['Mục đích']}")
                    st.write(f"**Kiểu xe:** {row['Kiểu xe']}")
                    st.write(f"**Tình trạng:** {row['Tình trạng']}")
                    st.write(f"**Hộp số:** {row['Hộp số']}")

                with col_score:
                    st.metric(
                        "Điểm đáp ứng",
                        f"{row['Điểm đáp ứng (%)']:.1f}%",
                    )
                    st.caption(f"Khoảng cách KNN: {row['Khoảng cách KNN']:.4f}")

                st.markdown("**Mức độ đáp ứng từng tiêu chí**")

                for col, score in row["Chi tiết tiêu chí"].items():
                    # Đã gọi hàm show_score_bar mới có định dạng màu sắc xanh/đỏ
                    show_score_bar(col, score)

                st.markdown("**Lý do đề xuất**")
                for explanation in row["Giải thích"]:
                    # Áp dụng màu cho phần chữ dựa vào dấu tích hoặc chéo
                    if "✓" in explanation:
                        st.markdown(f"<span style='color: #2e7d32; font-weight: bold;'>{explanation}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color: #d32f2f;'>{explanation}</span>", unsafe_allow_html=True)

with st.expander("Về hệ thống này"):
    st.markdown(
        """
        - Đây là hệ thống gợi ý dựa trên nội dung (**Content-based Recommendation**),
          sử dụng `NearestNeighbors` với khoảng cách Euclidean để tìm các xe
          gần nhất với nhu cầu người dùng.
        - Thuộc tính số (Giá, Công suất, Chỗ ngồi) được chuẩn hóa bằng biến đổi Logarit
          (Log-transform) và `MinMaxScaler` để giảm ảnh hưởng của các giá trị ngoại lệ.
        - Thuộc tính phân loại (Nhiên liệu, Mục đích, Tình trạng, Kiểu xe,
          Hộp số) được mã hóa bằng One-Hot Encoding.
        - Mỗi thuộc tính có trọng số riêng. Trọng số càng lớn thì thuộc tính
          càng ảnh hưởng đến khoảng cách KNN.
        - Tiêu chí người dùng chọn là "Không yêu cầu" sẽ không tham gia
          khoảng cách KNN.
        - KNN quyết định thứ tự Top-K. "Điểm đáp ứng (%)" và điểm từng tiêu chí
          được dùng để giải thích kết quả cho người dùng; đây là chỉ số hỗ trợ
          ra quyết định, không phải xác suất.
        """
    )