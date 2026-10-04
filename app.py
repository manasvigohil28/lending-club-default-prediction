"""Loan Default Prediction dashboard.

Run with:   streamlit run app.py

The dashboard does not train anything. It loads the files saved by Section 11 of
loan_default_prediction.ipynb, so run the notebook once before starting it:
  - data/loans_clean.csv      cleaned loans (for the charts)
  - models/loan_models.pkl    trained models, scaler, feature list and test results
"""
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

st.set_page_config(page_title='Loan Default Prediction', page_icon='💳', layout='wide')

# Same colours as the notebook charts (colour-blind safe)
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#898781'
STATUS_COLORS = {'Fully Paid': BLUE, 'Charged Off': ORANGE}
MODEL_COLORS = {'Logistic Regression': BLUE, 'Random Forest': ORANGE, 'XGBoost': AQUA}

GRADE_MAP = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
CATEGORY_COLS = ['home_ownership', 'verification_status', 'purpose', 'application_type']

PAGES = ['Overview', 'Explore the Data', 'Model Performance', 'Predict a Loan', 'Approval Strategy']


# ------------------------------------------------------------------ load data and models
@st.cache_data
def load_data():
    return pd.read_csv('data/loans_clean.csv')


@st.cache_resource
def load_models():
    return joblib.load('models/loan_models.pkl')


if not (os.path.exists('data/loans_clean.csv') and os.path.exists('models/loan_models.pkl')):
    st.error('Saved files not found. Run loan_default_prediction.ipynb once (Run All), then restart the dashboard.')
    st.stop()

df = load_data()
bundle = load_models()
models, results = bundle['models'], bundle['results']
BASE_RATE = df['default'].mean() * 100


def style(fig, height=360):
    """Give every Plotly chart the same clean look."""
    fig.update_layout(template='plotly_white', height=height, margin=dict(l=10, r=10, t=75, b=10),
                      legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0, title=None),
                      font=dict(size=13))
    if fig.layout.title.text:
        fig.update_layout(title=dict(y=0.98, yanchor='top'))
    fig.update_traces(textposition='outside', cliponaxis=False, selector=dict(type='bar'))
    return fig


def default_rate_by(data, col):
    """Default rate (%) and number of loans for each value of a column."""
    out = data.groupby(col)['default'].agg(['mean', 'count']).reset_index()
    out['Default rate (%)'] = out['mean'] * 100
    return out.rename(columns={'count': 'Loans'})


# ------------------------------------------------------------------ sidebar
st.sidebar.title('💳 Loan Default Prediction')
st.sidebar.caption('Lending Club loans, 2007–2018')
start_page = st.query_params.get('page', 'Overview')
page = st.sidebar.radio('Go to', PAGES, index=PAGES.index(start_page) if start_page in PAGES else 0)
st.sidebar.divider()
st.sidebar.markdown(f'**Data:** {len(df):,} finished loans  \n**Best model:** {bundle["best_model"]}  \n'
                    f'**Test ROC-AUC:** {results.loc[bundle["best_model"], "ROC-AUC"]:.3f}')


# ================================================================== 1. Overview
if page == 'Overview':
    st.title('Loan Default Overview')
    st.caption('A random sample of 100,000 finished Lending Club loans (Fully Paid or Charged Off), after cleaning.')

    f1, f2, f3 = st.columns(3)
    terms = f1.multiselect('Loan term (months)', sorted(df['term'].unique()), default=sorted(df['term'].unique()))
    grades = f2.multiselect('Grade', sorted(df['grade'].unique()), default=sorted(df['grade'].unique()))
    homes = f3.multiselect('Home ownership', sorted(df['home_ownership'].unique()),
                           default=sorted(df['home_ownership'].unique()))
    data = df[df['term'].isin(terms) & df['grade'].isin(grades) & df['home_ownership'].isin(homes)]
    if data.empty:
        st.warning('No loans match these filters.')
        st.stop()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Loans', f'{len(data):,}')
    difference = data['default'].mean() * 100 - BASE_RATE
    k2.metric('Default rate', f'{data["default"].mean() * 100:.1f}%', delta_color='inverse',
              delta=f'{difference:+.1f} pts vs all loans' if abs(difference) >= 0.05 else None)
    k3.metric('Average loan', f'${data["loan_amnt"].mean():,.0f}')
    k4.metric('Average interest rate', f'{data["int_rate"].mean():.1f}%')

    c1, c2 = st.columns(2)
    status = data['loan_status'].value_counts().reindex(['Fully Paid', 'Charged Off']).reset_index()
    fig = px.pie(status, names='loan_status', values='count', hole=0.55, color='loan_status',
                 color_discrete_map=STATUS_COLORS, title='Loan status')
    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='white', width=2)))
    c1.plotly_chart(style(fig).update_layout(showlegend=False))

    rate = default_rate_by(data, 'grade')
    fig = px.bar(rate, x='grade', y='Default rate (%)', text_auto='.1f', title='Default rate by grade',
                 hover_data={'Loans': ':,'}, color_discrete_sequence=[BLUE])
    c2.plotly_chart(style(fig).update_layout(xaxis_title='Grade (A = safest, G = riskiest)'))

    c3, c4 = st.columns(2)
    rate = default_rate_by(data, 'term')
    rate['term'] = rate['term'].astype(str) + ' months'
    fig = px.bar(rate, x='term', y='Default rate (%)', text_auto='.1f', title='Default rate by loan term',
                 hover_data={'Loans': ':,'}, color_discrete_sequence=[BLUE])
    c3.plotly_chart(style(fig).update_layout(xaxis_title=None))

    rate = default_rate_by(data, 'purpose')
    rate = rate[rate['Loans'] >= 100].sort_values('Default rate (%)')
    rate['purpose'] = rate['purpose'].str.replace('_', ' ')
    fig = px.bar(rate, x='Default rate (%)', y='purpose', orientation='h', text_auto='.1f',
                 title='Default rate by loan purpose (at least 100 loans)', hover_data={'Loans': ':,'},
                 color_discrete_sequence=[BLUE])
    c4.plotly_chart(style(fig).update_layout(yaxis_title=None))


# ================================================================== 2. Explore the data
elif page == 'Explore the Data':
    st.title('Explore the Data')
    st.caption('Pick any feature to see how it relates to default.')

    category_options = {'Grade': 'grade', 'Loan term': 'term', 'Home ownership': 'home_ownership',
                        'Income verification': 'verification_status', 'Employment length (years)': 'emp_length',
                        'Application type': 'application_type', 'Loan purpose': 'purpose'}
    numeric_options = {'Interest rate (%)': 'int_rate', 'FICO score': 'fico_score', 'Debt-to-income ratio': 'dti',
                       'Annual income ($)': 'annual_inc', 'Loan amount ($)': 'loan_amnt',
                       'Loan-to-income ratio': 'loan_to_income', 'Credit history (years)': 'credit_history_years',
                       'Revolving utilisation (%)': 'revol_util'}

    c1, c2 = st.columns(2)
    label = c1.selectbox('Category feature', list(category_options))
    col = category_options[label]
    rate = default_rate_by(df, col)
    rate[col] = rate[col].astype(str)
    fig = px.bar(rate, x=col, y='Default rate (%)', text_auto='.1f', hover_data={'Loans': ':,'},
                 title=f'Default rate by {label.lower()}', color_discrete_sequence=[BLUE])
    fig.add_hline(y=BASE_RATE, line_dash='dash', line_color=GREY, annotation_text=f'all loans {BASE_RATE:.1f}%',
                  annotation_position='top left')
    c1.plotly_chart(style(fig, 400).update_layout(xaxis_title=None))

    label = c2.selectbox('Numeric feature', list(numeric_options))
    col = numeric_options[label]
    fig = px.box(df, x='loan_status', y=col, color='loan_status', color_discrete_map=STATUS_COLORS, points=False,
                 category_orders={'loan_status': ['Fully Paid', 'Charged Off']}, title=f'{label} by loan status')
    c2.plotly_chart(style(fig, 400).update_layout(showlegend=False, xaxis_title=None, yaxis_title=label))

    medians = df.groupby('loan_status')[list(numeric_options.values())].median().T
    medians.index = list(numeric_options)
    st.subheader('Median values: defaulters vs non-defaulters')
    st.dataframe(medians[['Charged Off', 'Fully Paid']].style.format('{:,.2f}'))

    corr = df.select_dtypes('number').corr()['default'].drop('default').sort_values()
    fig = px.bar(x=corr.values, y=corr.index, orientation='h', title='Correlation of each feature with default',
                 color=np.where(corr.values > 0, 'Higher risk', 'Lower risk'),
                 color_discrete_map={'Higher risk': ORANGE, 'Lower risk': BLUE})
    st.plotly_chart(style(fig, 520).update_layout(xaxis_title='Correlation with default', yaxis_title=None))


# ================================================================== 3. Model performance
elif page == 'Model Performance':
    st.title('Model Performance')
    st.caption('All three models were trained on 80% of the loans and tested on the other 20% (19,994 loans).')

    st.dataframe(results.style.format('{:.3f}').highlight_max(subset=['CV ROC-AUC', 'Recall', 'F1', 'ROC-AUC'],
                                                              color='#dce8f7'))

    c1, c2 = st.columns(2)
    long = results[['Precision', 'Recall', 'F1', 'ROC-AUC']].reset_index().melt(id_vars='Model', var_name='Metric')
    fig = px.bar(long, x='Metric', y='value', color='Model', barmode='group', text_auto='.2f',
                 color_discrete_map=MODEL_COLORS, title='Metric comparison')
    fig.update_traces(textangle=0, textposition='outside', textfont_size=11)
    c1.plotly_chart(style(fig, 420).update_layout(yaxis_title='Score', yaxis_range=[0, 1], xaxis_title=None))

    fig = go.Figure()
    for name, prob in bundle['probs'].items():
        fpr, tpr, _ = roc_curve(bundle['y_test'], prob)
        step = max(1, len(fpr) // 400)   # fewer points keeps the chart fast
        fig.add_scatter(x=fpr[::step], y=tpr[::step], mode='lines', line=dict(color=MODEL_COLORS[name], width=2),
                        name=f'{name} (AUC {results.loc[name, "ROC-AUC"]:.3f})')
    fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color=GREY, dash='dash', width=1), name='Random guess')
    fig.update_layout(title='ROC curves', xaxis_title='False positive rate', yaxis_title='True positive rate (recall)')
    style(fig, 420).update_layout(legend=dict(orientation='v', x=0.98, y=0.04, xanchor='right', yanchor='bottom'))
    c2.plotly_chart(fig)

    c3, c4 = st.columns(2)
    name = c3.selectbox('Confusion matrix for', list(models), index=list(models).index(bundle['best_model']))
    cm = confusion_matrix(bundle['y_test'], (bundle['probs'][name] >= 0.5).astype(int))
    fig = px.imshow(cm, text_auto=',', color_continuous_scale=['#cde2fb', '#0d366b'],
                    x=['Fully Paid', 'Charged Off'], y=['Fully Paid', 'Charged Off'],
                    labels=dict(x='Predicted', y='Actual', color='Loans'), title=f'Confusion matrix: {name}')
    c3.plotly_chart(style(fig, 420).update_layout(coloraxis_showscale=False))
    tn, fp, fn, tp = cm.ravel()
    c3.caption(f'Caught {tp:,} of {tp + fn:,} defaulters ({tp / (tp + fn):.0%}); '
               f'wrongly flagged {fp:,} of {tn + fp:,} good borrowers ({fp / (tn + fp):.0%}).')

    top = bundle['importance'].sort_values().tail(15)
    fig = px.bar(x=top.values, y=top.index, orientation='h', title='Top 15 features (XGBoost importance)',
                 color_discrete_sequence=[BLUE])
    c4.plotly_chart(style(fig, 470).update_layout(xaxis_title='Importance', yaxis_title=None))


# ================================================================== 4. Predict a loan
elif page == 'Predict a Loan':
    st.title('Predict a Loan')
    st.caption('Enter the details of a new application. The starting values are typical (median) values from the data.')

    def median(col):
        return float(df[col].median())

    left, right = st.columns([2.4, 1], gap='large')
    with left, st.form('applicant'):
        c1, c2, c3 = st.columns(3)
        c1.subheader('Loan')
        loan_amnt = c1.number_input('Loan amount ($)', 500, 40000, 12000, step=500)
        term = c1.selectbox('Term (months)', [36, 60])
        int_rate = c1.slider('Interest rate (%)', 5.0, 31.0, round(median('int_rate'), 1), 0.1)
        grade = c1.selectbox('Grade', list(GRADE_MAP), index=1)
        purpose = c1.selectbox('Purpose', sorted(df['purpose'].unique()),
                               index=sorted(df['purpose'].unique()).index('debt_consolidation'))

        c2.subheader('Borrower')
        annual_inc = c2.number_input('Annual income ($)', 5000, 500000, int(median('annual_inc')), step=1000)
        emp_length = c2.slider('Years in current job', 0, 10, int(median('emp_length')))
        home_ownership = c2.selectbox('Home ownership', ['MORTGAGE', 'RENT', 'OWN'])
        verification_status = c2.selectbox('Income verification', ['Not Verified', 'Source Verified', 'Verified'])
        application_type = c2.selectbox('Application type', ['Individual', 'Joint App'])

        c3.subheader('Credit history')
        fico_score = c3.slider('FICO score', 660, 850, int(median('fico_score')))
        dti = c3.slider('Debt-to-income ratio', 0.0, 40.0, round(median('dti'), 1), 0.1)
        credit_history_years = c3.slider('Years since first credit line', 3, 60, int(median('credit_history_years')))
        revol_util = c3.slider('Credit card utilisation (%)', 0, 120, int(median('revol_util')))
        with c3.expander('More credit details'):
            revol_bal = st.number_input('Credit card balance ($)', 0, 100000, int(median('revol_bal')), step=500)
            open_acc = st.number_input('Open credit lines', 0, 60, int(median('open_acc')))
            total_acc = st.number_input('Total credit lines ever', 1, 120, int(median('total_acc')))
            mort_acc = st.number_input('Mortgage accounts', 0, 30, int(median('mort_acc')))
            inq_last_6mths = st.number_input('Credit enquiries (last 6 months)', 0, 10, 0)
            delinq_2yrs = st.number_input('Late payments (last 2 years)', 0, 20, 0)
            pub_rec = st.number_input('Public records', 0, 20, 0)
            pub_rec_bankruptcies = st.number_input('Bankruptcies', 0, 10, 0)

        model_name = st.selectbox('Model', list(models), index=list(models).index(bundle['best_model']))
        st.form_submit_button('Predict default risk', type='primary')

    # Monthly payment from the standard loan formula
    r = int_rate / 100 / 12
    installment = loan_amnt * r / (1 - (1 + r) ** -term)
    # The models were trained on capped income, DTI and balance (notebook Section 3.6), so cap the inputs the same way
    annual_inc_capped = float(np.clip(annual_inc, df['annual_inc'].min(), df['annual_inc'].max()))

    applicant = {
        'loan_amnt': loan_amnt, 'term': term, 'int_rate': int_rate, 'installment': installment, 'grade': grade,
        'emp_length': emp_length, 'home_ownership': home_ownership, 'annual_inc': annual_inc_capped,
        'verification_status': verification_status, 'purpose': purpose,
        'dti': float(np.clip(dti, df['dti'].min(), df['dti'].max())), 'delinq_2yrs': delinq_2yrs,
        'inq_last_6mths': inq_last_6mths, 'open_acc': open_acc, 'pub_rec': pub_rec,
        'revol_bal': float(np.clip(revol_bal, df['revol_bal'].min(), df['revol_bal'].max())),
        'revol_util': revol_util, 'total_acc': total_acc, 'mort_acc': mort_acc,
        'pub_rec_bankruptcies': pub_rec_bankruptcies, 'application_type': application_type,
        'fico_score': fico_score, 'credit_history_years': credit_history_years,
        'loan_to_income': loan_amnt / annual_inc_capped,
    }

    # Same encoding as the notebook: grade A-G -> 1-7, one-hot categories, same 38 columns, same scaler
    row = pd.DataFrame([applicant])
    row['grade'] = row['grade'].map(GRADE_MAP)
    row = pd.get_dummies(row, columns=CATEGORY_COLS, dtype=int)
    row = row.reindex(columns=bundle['feature_columns'], fill_value=0)
    prob = models[model_name].predict_proba(bundle['scaler'].transform(row))[0, 1] * 100

    with right:
        fig = go.Figure(go.Indicator(
            mode='gauge+number', value=prob, number={'suffix': '%', 'valueformat': '.1f'},
            title={'text': 'Model risk score'},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#0b0b0b', 'thickness': 0.25},
                   'steps': [{'range': [0, 35], 'color': '#cde2fb'}, {'range': [35, 50], 'color': '#fbd9c8'},
                             {'range': [50, 100], 'color': '#f4a582'}],
                   'threshold': {'line': {'color': ORANGE, 'width': 3}, 'value': 50}}))
        st.plotly_chart(style(fig, 250).update_layout(margin=dict(t=60, b=0)))

        if prob >= 50:
            st.error('**High risk: flag for review**  \nThe score is above the 50% cut-off used by the model.')
        elif prob >= 35:
            st.warning('**Medium risk**  \nClose to the 50% cut-off; worth a closer look.')
        else:
            st.success('**Low risk: approve**  \nThe score is well below the 50% cut-off.')

        m1, m2 = st.columns(2)
        m1.metric('Monthly payment', f'${installment:,.0f}')
        m2.metric('Loan-to-income', f'{applicant["loan_to_income"]:.0%}')
        st.markdown('**Real default rate in the data**')
        m3, m4 = st.columns(2)
        m3.metric(f'Grade {grade}', f'{df.loc[df["grade"] == grade, "default"].mean() * 100:.1f}%')
        m4.metric(f'{term} months', f'{df.loc[df["term"] == term, "default"].mean() * 100:.1f}%')
        st.caption('The models were trained with class weights, so the risk score is higher than the true chance of '
                   'default. It is best used to compare and rank applications.')


# ================================================================== 5. Approval strategy
elif page == 'Approval Strategy':
    st.title('Loan Approval Strategy')
    st.caption('Sort the 19,994 test loans from riskiest to safest using the model, and reject the riskiest share.')

    c1, c2 = st.columns([2, 1])
    reject_pct = c1.slider('Reject the riskiest share of applications (%)', 0, 60, 30, 5)
    model_name = c2.selectbox('Model used to rank loans', list(models), index=list(models).index(bundle['best_model']))

    y_true = bundle['y_test']
    y_sorted = y_true[np.argsort(-bundle['probs'][model_name])]
    k = int(len(y_sorted) * reject_pct / 100)
    rejected, approved = y_sorted[:k], y_sorted[k:]
    stopped = rejected.sum() / y_sorted.sum() * 100
    good_rejected = (rejected == 0).sum() / (y_sorted == 0).sum() * 100
    approved_rate = approved.mean() * 100 if len(approved) else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Loans rejected', f'{k:,}')
    m2.metric('Defaulters stopped', f'{stopped:.1f}%', help='Share of all real defaulters that were rejected')
    m3.metric('Good borrowers rejected', f'{good_rejected:.1f}%', help='Share of good borrowers turned away')
    m4.metric('Default rate of approved loans', f'{approved_rate:.1f}%',
              delta=f'{approved_rate - y_true.mean() * 100:+.1f} pts vs approving all', delta_color='inverse')

    rejected_share = np.arange(1, len(y_sorted) + 1) / len(y_sorted) * 100
    stopped_share = np.cumsum(y_sorted) / y_sorted.sum() * 100
    step = 20
    fig = go.Figure()
    fig.add_scatter(x=rejected_share[::step], y=stopped_share[::step], mode='lines', line=dict(color=BLUE, width=2),
                    name=f'{model_name} ranking')
    fig.add_scatter(x=[0, 100], y=[0, 100], mode='lines', line=dict(color=GREY, dash='dash', width=1),
                    name='Random rejection')
    if k > 0:
        fig.add_scatter(x=[reject_pct], y=[stopped], mode='markers', name='Your cut-off',
                        marker=dict(color=ORANGE, size=13, line=dict(color='white', width=2)))
    fig.update_layout(title='Defaulters stopped vs loans rejected', xaxis_title='Loans rejected (%), riskiest first',
                      yaxis_title='Defaulters stopped (%)')
    st.plotly_chart(style(fig, 420))

    rows = []
    for pct in [10, 20, 30, 40, 50]:
        n = int(len(y_sorted) * pct / 100)
        rows.append({'Reject riskiest': f'{pct}%',
                     'Defaulters stopped (%)': y_sorted[:n].sum() / y_sorted.sum() * 100,
                     'Good borrowers rejected (%)': (y_sorted[:n] == 0).sum() / (y_sorted == 0).sum() * 100,
                     'Default rate of approved loans (%)': y_sorted[n:].mean() * 100,
                     'Better than random (x)': (y_sorted[:n].sum() / y_sorted.sum()) / (pct / 100)})
    st.dataframe(pd.DataFrame(rows).set_index('Reject riskiest').style.format('{:.1f}'))
