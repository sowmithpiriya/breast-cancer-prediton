import pandas as pd
import matplotlib.pyplot as plt


def get_clean_data():
    # Load the UCI dataset (wdbc.data) from project root
    df = pd.read_csv('./wdbc.data', header=None)

    # Define proper column names based on UCI dataset
    columns = [
        'ID', 'Diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
        'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
        'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
        'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
        'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
        'fractal_dimension_worst'
    ]
    df.columns = columns

    # Drop the ID column (not useful for prediction)
    df = df.drop(['ID'], axis=1)

    # Convert Diagnosis to binary values: M = 1 (malignant), B = 0 (benign)
    df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

    # Rename to lowercase for consistency
    df.rename(columns={'Diagnosis': 'diagnosis'}, inplace=True)

    return df


def plot_data(df):
    plot = df['diagnosis'].value_counts().plot(
        kind='bar',
        title="Class distributions \n(0: Benign | 1: Malignant)",
        color=['#36B37E', '#FF5630']
    )
    plot.set_xlabel("Diagnosis")
    plot.set_ylabel("Frequency")
    plt.show()


def get_model():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    df = get_clean_data()

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis']

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train logistic regression
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    return model, scaler


def create_radar_chart(input_data):
    import plotly.graph_objects as go
    input_data = get_scaled_values_dict(input_data)

    fig = go.Figure()

    # Mean
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave_points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
               'Symmetry', 'Fractal Dimension'],
        fill='toself',
        name='Mean'
    ))

    # Standard Error
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave_points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
               'Symmetry', 'Fractal Dimension'],
        fill='toself',
        name='Standard Error'
    ))

    # Worst
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave_points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
               'Symmetry', 'Fractal Dimension'],
        fill='toself',
        name='Worst'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        autosize=True
    )
    return fig



def create_input_form(data):
    import streamlit as st

    st.sidebar.header("Cell Nuclei Details")

    slider_labels = [
        ("Radius (mean)", "radius_mean"), ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"), ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"), ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"), ("Concave points (mean)", "concave_points_mean"),
        ("Symmetry (mean)", "symmetry_mean"), ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"), ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"), ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"), ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"), ("Concave points (se)", "concave_points_se"),
        ("Symmetry (se)", "symmetry_se"), ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"), ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"), ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"), ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"), ("Concave points (worst)", "concave_points_worst"),
        ("Symmetry (worst)", "symmetry_worst"), ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]

    input_data = {}
    for label, col in slider_labels:
        input_data[col] = st.sidebar.slider(
            label, float(data[col].min()), float(data[col].max()), float(data[col].mean()))
    return input_data



def get_scaled_values_dict(values_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}
    for key, value in values_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_dict[key] = (value - min_val) / (max_val - min_val)
    return scaled_dict



def display_predictions(input_data, model, scaler):
    import streamlit as st
    import numpy as np

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_data_scaled = scaler.transform(input_array)

    st.subheader('Cell cluster prediction')
    st.write("The cell cluster is:")

    prediction = model.predict(input_data_scaled)
    if prediction[0] == 0:
        st.success("Benign (0)")
    else:
        st.error("Malignant (1)")

    st.write("**Probability of being benign:**",
             model.predict_proba(input_data_scaled)[0][0])
    st.write("**Probability of being malignant:**",
             model.predict_proba(input_data_scaled)[0][1])
    st.caption("⚠️ This app assists diagnosis but is not a substitute for professional medical advice.")



def create_app():
    import streamlit as st

    st.set_page_config(page_title="Breast Cancer Diagnosis",
                       page_icon=":female-doctor:", layout="wide", initial_sidebar_state="expanded")

    # Load CSS
    with open("./assets/style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    with st.container():
        st.title("Breast Cancer Diagnosis")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer from your tissue sample. "
                 "This app predicts using a machine learning model whether a breast mass is benign or malignant "
                 "based on the measurements it receives from your cytology lab. "
                 "You can also update the measurements by hand using the sliders in the sidebar.")

    data = get_clean_data()
    input_data = create_input_form(data)

    model, scaler = get_model()

    col1, col2 = st.columns([4, 1])
    with col1:
        radar_chart = create_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)
    with col2:
        display_predictions(input_data, model, scaler)



def main():
    create_app()


if __name__ == '__main__':
    main()
