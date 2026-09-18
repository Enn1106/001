
import streamlit as st
import pandas as pd

from recommender import load_and_clean_data, CarRecommender, DEFAULT_WEIGHTS

st.set_page_config(page_title="Gợi ý xe ô tô theo nhu cầu", layout="wide")


@st.cache_data
def get_data():
    return load_and_clean_data("cars.csv")


@st.cache_resource
def get_recommender(_df: pd.DataFrame, weights: dict):
    return CarRecommender(_df, weights=weights)


def format_vnd(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".") + " VNĐ"


df = get_data()

st.title("Hệ thống hỗ trợ lựa chọn xe ô tô theo nhu cầu sử dụng")
st.caption("Nhóm 8 · Đồ án môn Hệ trợ giúp quyết định — Gợi ý xe bằng thuật toán K-Nearest Neighbors (KNN)")

with st.sidebar:
    st.header(" Trọng số thuộc tính")
    st.caption("Điều chỉnh mức độ ưu tiên khi tính độ tương đồng giữa các xe.")
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
st.caption("Bỏ trống / chọn 'Không yêu cầu' nếu tiêu chí đó không quan trọng với bạn.")

col1, col2, col3, col4 = st.columns(4)

with col1:
    use_price = st.checkbox("Có yêu cầu về giá", value=True)
    # Hiển thị thanh trượt theo đơn vị "triệu VNĐ" cho số ngắn, dễ nhìn hơn
    # (thay vì phải hiện cả dãy số 000.000 dài).
    price_trieu = st.slider(
        "Ngân sách (VNĐ)",
        min_value=int(df["Giá( VNĐ)"].min() / 1_000_000),
        max_value=int(df["Giá( VNĐ)"].max() / 1_000_000),
        value=int(df["Giá( VNĐ)"].median() / 1_000_000),
        step=10,
        disabled=not use_price,
    )
    price = price_trieu * 1_000_000
    if use_price:
        st.caption(f"≈ {format_vnd(price)}")

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
    seats_options = ["Không yêu cầu"] + [str(s) for s in recommender.unique_values("Chỗ ngồi")]
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

    if not query:
        st.warning("Vui lòng nhập ít nhất một tiêu chí để hệ thống có thể gợi ý.")
    else:
        result = recommender.recommend(query, k=int(k))

        st.subheader("2. Kết quả gợi ý")
        st.caption(f"Top {len(result)} xe gần nhất với tiêu chí bạn nhập (theo khoảng cách KNN có trọng số)")

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
                "Độ phù hợp (%)",
            ]
        ].copy()
        display_df["Giá( VNĐ)"] = display_df["Giá( VNĐ)"].apply(format_vnd)

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.bar_chart(result.set_index("Tên xe")["Độ phù hợp (%)"])

with st.expander(" Về hệ thống này"):
    st.markdown(
        """
        - Đây là hệ gợi ý dựa trên nội dung (**content-based recommendation**), dùng
          `NearestNeighbors` (KNN không giám sát) để tìm những xe có đặc điểm **gần giống nhất**
          với tiêu chí khách hàng nhập vào.
        - Thuộc tính số (Giá, Công suất, Chỗ ngồi) được chuẩn hóa Min-Max; thuộc tính phân loại
          (Nhiên liệu, Mục đích, Tình trạng, Kiểu xe, Hộp số) được mã hóa One-Hot.
        - Mỗi thuộc tính có **trọng số** riêng khi tính khoảng cách — điều chỉnh ở thanh bên trái.
        - **Độ phù hợp (%)** được tính dựa trên khoảng cách so với xe xa nhất trong toàn bộ dữ liệu.
        """
    )