import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

cat_cols = ["State", "City", "Property_Type", "Furnished_Status", "Facing", "Owner_Type",
            "Availability_Status", "Public_Transport_Accessibility", "Parking_Space", "Security"]
num_cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_Rs", "Age_of_Property",
            "Nearby_Schools", "Nearby_Hospitals", "Floor_No", "Total_Floors"]

@st.cache_resource
def load_models():
    return (joblib.load("good_investment_model.pkl"), joblib.load("future_price_model.pkl"),
            joblib.load("classifier_columns.pkl"), joblib.load("regressor_columns.pkl"))

@st.cache_data
def load_data():
    return pd.read_csv("processed_housing_data.csv.gz")

clf, reg, clf_cols, reg_cols = load_models()
df = load_data()

def build_row(inputs, cols):
    row = pd.get_dummies(pd.DataFrame([inputs])[num_cols + cat_cols], columns=cat_cols)
    return row.reindex(columns=cols, fill_value=0).astype(float)

st.title("🏠 Real Estate Investment Advisor")
st.caption("Predicts if a property is a Good Investment and its estimated price after 5 years.")

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "🔎 Explore Properties", "📊 Market Insights"])

# ---------------- TAB 1: PREDICT ----------------
with tab1:
    st.subheader("Enter property details")
    c1, c2, c3 = st.columns(3)
    with c1:
        state = c1.selectbox("State", sorted(df["State"].unique()))
        city = c1.selectbox("City", sorted(df[df["State"] == state]["City"].unique()))
        ptype = c1.selectbox("Property Type", sorted(df["Property_Type"].unique()))
        bhk = c1.slider("BHK", 1, 5, 3)
        size = c1.number_input("Size (SqFt)", 500, 5000, 2000, step=50)
    with c2:
        price = c2.number_input("Price (Lakhs)", 10.0, 500.0, 250.0, step=5.0)
        age = c2.slider("Age of Property (years)", 2, 35, 15)
        total_floors = c2.number_input("Total Floors", 1, int(df["Total_Floors"].max()), 10)
        floor_no = c2.number_input("Floor No", 0, int(total_floors), min(3, int(total_floors)))
        furnished = c2.selectbox("Furnished Status", sorted(df["Furnished_Status"].unique()))
    with c3:
        avail = c3.selectbox("Availability", sorted(df["Availability_Status"].unique()))
        transport = c3.selectbox("Public Transport", ["Low", "Medium", "High"])
        schools = c3.slider("Nearby Schools", int(df["Nearby_Schools"].min()), int(df["Nearby_Schools"].max()), 5)
        hospitals = c3.slider("Nearby Hospitals", int(df["Nearby_Hospitals"].min()), int(df["Nearby_Hospitals"].max()), 5)
        facing = c3.selectbox("Facing", sorted(df["Facing"].unique()))
        owner = c3.selectbox("Owner Type", sorted(df["Owner_Type"].unique()))
        parking = c3.selectbox("Parking Space", ["Yes", "No"])
        security = c3.selectbox("Security", ["Yes", "No"])

    if st.button("Analyze Property", type="primary"):
        inputs = {"State": state, "City": city, "Property_Type": ptype, "BHK": bhk,
                  "Size_in_SqFt": size, "Price_in_Lakhs": price,
                  "Price_per_SqFt_Rs": round(price * 100000 / size), "Age_of_Property": age,
                  "Furnished_Status": furnished, "Floor_No": floor_no, "Total_Floors": total_floors,
                  "Nearby_Schools": schools, "Nearby_Hospitals": hospitals,
                  "Public_Transport_Accessibility": transport, "Parking_Space": parking,
                  "Security": security, "Facing": facing, "Owner_Type": owner,
                  "Availability_Status": avail}

        good = int(clf.predict(build_row(inputs, clf_cols))[0])
        prob = float(clf.predict_proba(build_row(inputs, clf_cols))[0][1])
        future = float(reg.predict(build_row(inputs, reg_cols))[0])

        r1, r2, r3 = st.columns(3)
        if good:
            r1.success("✅ Good Investment")
        else:
            r1.error("❌ Not a Good Investment")
        r1.metric("Model confidence (Good)", f"{prob*100:.1f}%")
        r2.metric("Estimated price after 5 years", f"₹ {future:,.1f} Lakhs")
        r3.metric("Expected gain", f"₹ {future - price:,.1f} Lakhs",
                  f"{(future/price - 1)*100:.1f}% total")

        st.subheader("Feature importance (Classifier)")
        imp = pd.Series(clf.feature_importances_, index=clf_cols).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=imp.values, y=imp.index, ax=ax)
        st.pyplot(fig)

# ---------------- TAB 2: EXPLORE ----------------
with tab2:
    st.subheader("Filter properties")
    f1, f2, f3, f4 = st.columns(4)
    states = f1.multiselect("State", sorted(df["State"].unique()))
    bhks = f2.multiselect("BHK", sorted(df["BHK"].unique()))
    prange = f3.slider("Price (Lakhs)", 10, 500, (10, 500))
    srange = f4.slider("Size (SqFt)", 500, 5000, (500, 5000))

    fdf = df[df["Price_in_Lakhs"].between(*prange) & df["Size_in_SqFt"].between(*srange)]
    if states:
        fdf = fdf[fdf["State"].isin(states)]
    if bhks:
        fdf = fdf[fdf["BHK"].isin(bhks)]

    st.write(f"**{len(fdf):,}** properties found | Good Investments: **{fdf['Good_Investment'].mean()*100:.1f}%**")
    st.dataframe(fdf[["State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
                      "Price_in_Lakhs", "Availability_Status", "Good_Investment", "Future_Price_5Y"]].head(200))

# ---------------- TAB 3: INSIGHTS ----------------
with tab3:
    st.subheader("Average price by city")
    city_price = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(15)
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    sns.barplot(x=city_price.values, y=city_price.index, ax=ax1)
    ax1.set_xlabel("Avg Price (Lakhs)")
    st.pyplot(fig1)

    st.subheader("Heatmap: Avg price per SqFt (Rs) by State and Property Type")
    pivot = df.pivot_table(index="State", columns="Property_Type", values="Price_per_SqFt_Rs", aggfunc="mean")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Price trend by year built")
    trend = df.groupby("Year_Built")["Price_per_SqFt_Rs"].mean()
    st.line_chart(trend)

    st.info("Note: This dataset is synthetic. Future price uses an assumed growth rate (about 7-10% per year), "
            "and the Good Investment label is rule-based.")
