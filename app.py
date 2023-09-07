
from operator import index
import streamlit as st
import sys
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os

if os.path.exists('./sourcedata.csv'):
    df_csv = pd.read_csv('sourcedata.csv', index_col=None)


with st.sidebar:
    st.title("AutoML")
    st.info("By Akshith Chowdary")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTnxi_YIKmP9dAESxenlz-fl5En_rFFSgdYtw&usqp=CAU", width=250)
    choose = st.radio("Options", ['Upload', 'Profiling', 'Model Building', 'Download Model'])
    st.info("This application will help you to explore your data, build and train a ML Model, save your model.")

if choose == "Upload":
    st.title("Upload Your CSV Dataset")
    data_uploaded_csv = st.file_uploader("Upload Your CSV Dataset")
    if data_uploaded_csv:
        df_csv = pd.read_csv(data_uploaded_csv, index_col=None, encoding='ISO-8859-1')
        df_csv.to_csv('sourcedata.csv', index=None)
        st.dataframe(df_csv, width=900)

    st.title("Upload Your Excel Dataset")
    data_uploaded_excel = st.file_uploader("Upload Your EXCEL Dataset")
    if data_uploaded_excel:
        df_excel = pd.read_excel(data_uploaded_excel, index_col=None)
        df_excel.to_csv('sourcedata.csv', index=None)
        df_csv = pd.read_csv('sourcedata.csv')
        st.dataframe(df_csv, width=900)

if choose == "Profiling":
    st.title("Profiling your Data")

    profile_df = df_csv.profile_report()
    st_profile_report(profile_df)
    profile_df = df_csv.profile_report()
    st_profile_report(profile_df)

if choose == "Model Building":

    option = st.selectbox("Select your problem type",
                          ("Select an option", "Regression", "Classification", "Anomaly Detection", "Clustering",
                           "Time Series", "Topic Modelling"))
    st.write('You selected:', option)

    if option == "Classification":
        from pycaret.classification import setup, compare_models, pull, save_model, load_model, plot_model, \
            predict_model

        options_input = st.multiselect("Choose the input features", df_csv.columns)
        option_target = st.selectbox("Choose the target features", df_csv.columns)
        if st.button("Train Model"):
            X = df_csv[options_input]
            y = df_csv[option_target]
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
            # clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            # models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            # st.dataframe(models)
            setup(data=X, target=y)

            #
            best_model_classification = compare_models()
            compare_df = pull(best_model_classification)

            st.dataframe(compare_df)
            st.info(best_model_classification)
            st.title("Analysis of model")
            plot_model(best_model_classification, plot='auc', display_format='streamlit')
            plot_model(best_model_classification, plot='confusion_matrix', display_format='streamlit')
            plot_model(best_model_classification, plot='class_report', display_format='streamlit')
            plot_model(best_model_classification, plot='pr', display_format='streamlit')
            save_model(best_model_classification, 'best_model_classification')
            st.title("Model Predictions")
            st.dataframe(predict_model(best_model_classification))
            predictions = predict_model(best_model_classification, data=df_csv, raw_score=True)
            #st.dataframe(predictions).head(20)

    if option == "Regression":
        from pycaret.regression import setup, compare_models, pull, save_model, load_model, plot_model, predict_model

        options_input = st.multiselect("Choose the input features", df_csv.columns)
        option_target = st.selectbox("Choose the target features", df_csv.columns)
        if st.button("Train Model"):
            X = df_csv[options_input]
            y = df_csv[option_target]
            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
            # clf = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            # models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            # st.dataframe(models)
            setup(data=X, target=y)

            #
            best_model_regression = compare_models()
            compare_df = pull(best_model_regression)
            st.dataframe(compare_df)
            st.info(best_model_regression)
            st.title("Analysis of the model")
            plot_model(best_model_regression, plot='residuals', display_format='streamlit')
            plot_model(best_model_regression, plot='error', display_format='streamlit')
            plot_model(best_model_regression, plot='vc', display_format='streamlit')
            save_model(best_model_regression, 'best_model_regression')
            st.title("Model Predictions")
            st.dataframe(predict_model(best_model_regression))
            predictions = predict_model(best_model_regression, data=df_csv)
            #st.dataframe(predictions).head(20)

    if option == "Anomaly Detection":
        from pycaret.anomaly import setup, create_model, models, plot_model, save_model, assign_model, predict_model

        if st.button("Train Model"):
            setup(data=df_csv)
            knn = create_model('knn')
            st.dataframe(models())
            st.info(knn)
            st.title("Analysis of model")
            plot_model(knn, plot='tsne', display_format='streamlit')
            plot_model(knn, plot='umap')
            save_model(knn, 'best_model_knn')
            result = assign_model(knn)
            st.dataframe(result)
            predictions = predict_model(knn, 'knn_pipeline', data=df_csv)
            st.dataframe(predictions).head(20)

    if option == "Clustering":
        from pycaret.clustering import setup, create_model, plot_model, save_model, predict_model, assign_model

        # option_input = st.selectbox("Choose the input feature", df_csv.columns)

        if st.button("Train Model"):
            setup(data=df_csv, normalize=True)

            best_model_kmeans = create_model('kmeans', random_state=42)
            st.info(best_model_kmeans)
            st.title("Analysis of model")
            plot_model(best_model_kmeans, plot='elbow', display_format='streamlit')
            plot_model(best_model_kmeans, plot='silhouette', display_format='streamlit')
            plot_model(best_model_kmeans, plot='cluster', display_format='streamlit')
            plot_model(best_model_kmeans, plot='distribution', display_format='streamlit')
            save_model(best_model_kmeans, 'best_model_kmeans')
            result = assign_model(best_model_kmeans)
            st.dataframe(result)
            predictions = predict_model(best_model_kmeans, data=df_csv)
            st.dataframe(predictions).head(20)

    if option == "Time Series":
        from pycaret.time_series import setup, compare_models, plot_model, pull, save_model, predict_model, \
            finalize_model

        option_target = st.selectbox("Choose the target features", df_csv.columns)
        if st.button("Train Model"):
            setup(data=df_csv, fh=3, fold=5, target=option_target)
            best_model_timeseries = compare_models()
            compare_df = pull(best_model_timeseries)
            st.dataframe(compare_df)
            st.info(best_model_timeseries)
            st.title("Analysis of model")
            plot_model(best_model_timeseries, plot='forecast', display_format='streamlit')
            plot_model(best_model_timeseries, plot='diagnostics', display_format='streamlit')
            plot_model(best_model_timeseries, plot='insample', display_format='streamlit')
            save_model(best_model_timeseries, 'best_model_timeseries')
            # finalize model
            final_best = finalize_model(best_model_timeseries)
            predictions = predict_model(best_model_timeseries, fh=24)
            st.dataframe(predictions).head(20)

    if option == "Topic Modelling":
        from pycaret.nlp import setup, models, plot_model, create_model, save_model, assign_model

        option_target = st.selectbox("Choose the target features", df_csv.columns)
        if st.button("Train Model"):
            setup(df_csv, target=option_target)
            st.dataframe(models())
            best_model_tm = create_model('lda')
            st.info(best_model_tm)
            st.title("Analysis of model")

            plot_model(best_model_tm, plot='frequency', display_format='streamlit')
            plot_model(best_model_tm, plot='sentiment', display_format='streamlit')
            plot_model(best_model_tm, plot='wordcloud', display_format='streamlit')
            save_model(best_model_tm, 'best_model_tm')
            lda_results = assign_model(best_model_tm)
            st.dataframe(lda_results).head(20)

if choose == "Download Model":
    with open('best_model_classification.pkl', 'rb') as f1:
        st.download_button('Download Model Classification', f1, file_name="best_model_classification.pkl")

    with open('best_model_regression.pkl', 'rb') as f2:
        st.download_button('Download Model Regression', f2, file_name="best_model_regression.pkl")

    with open('best_model_knn.pkl', 'rb') as f3:
        st.download_button('Download Model Anamoly Detection', f3, file_name="best_model_knn.pkl")

    with open('best_model_kmeans.pkl', 'rb') as f4:
        st.download_button('Download Model Clustering', f4, file_name="best_model_kmeans.pkl")

    with open('best_model_timeseries.pkl', 'rb') as f5:
        st.download_button('Download Model Time Series', f5, file_name="best_model_timeseries.pkl")

    with open('best_model_tm.pkl', 'rb') as f6:
        st.download_button('Download Model Topic Modelling', f6, file_name="best_model_tm.pkl")
