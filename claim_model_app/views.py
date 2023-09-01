import json
import pandas as pd
from rest_framework.response import Response
from rest_framework.views import APIView
# from rest_framework.response import Response
from django.db import connection
from django.conf import settings
from configparser import ConfigParser
from time import time
from .apps import ClaimModelAppConfig
import heapq
from collections import Counter
import itertools
import warnings

warnings.filterwarnings("ignore")
import logging
import os

config = ConfigParser()
config.read("config/config.ini")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename=os.path.join(settings.LOG_DIR, settings.LOG_FILE), level=logging.INFO)


def get_patient_details(self, patient_id):
    """
    this function will fetch the all records
    for the given patient id
    :param self:
    :param int patient_id:
    :return: a dataframe
    """

    cursor = connection.cursor()
    query = """
SELECT Top 1 t2.Voucher_ID, t2.Voucher_Number, t2.Claim_Number, t2.Original_Billing_Date, t2.Fees, t2.Posted_Payments,
t2.Posted_Adjustments, t2.Posted_Refunds, t2.CoPayment_Amount, t2.Patient_ID, t2.Patient_Policy_ID,
t2.Is_Self_Pay, t4.Service_ID, t4.Service_Date_From, t4.Original_Billing_Date,
t13.Date_Paid, t4.Patient_Number, t4.Original_Carrier_Name, t4.Original_Carrier_Category_Abbr,
t4.Actual_Dr_Name, t4.Place_of_Service_Descr, t4.Procedure_Code, t4.Service_Units, t4.Service_Fee, t13.Amount,
t13.Transaction_Code_Descr,
t4.Primary_Diagnosis_Code, t5.Patient_Control_Number, t5.Claim_Status,
t5.Total_Claim_Charge_Amount, t5.Claim_Payment_Amount, t5.Patient_Responsibility_Amount, t5.Claim_Frequency_Code,
t5.Claim_DRG_Amount, t5.Claim_PPS_Outlier_Amount, t5.Claim_Stmt_Period_Start,
t5.Claim_Stmt_Period_End, t5.Coverage_Expiration_Date, t5.Claim_Received_Date,
t7.Allowed, t7.Deductible, t7.CoPayment, t7.CoInsurance, t7.EOB_Date, t7.Reimbursement_Comment_ID,
t13.Reimbursement_Comment_Abbr, t7.Pending,
t7.Denied, t9.Coverage, t9.Waive_CoPayment, t9.Policy_ID, t9.Verified_Date, t10.Group_Name, t10.Certificate_No, t10.Comments,
t10.Effective_Date, t10.Expiration_Date, t11.Plan_Code,
t11.Description, t11.Is_Capitated_Plan, t12.Abbreviation,
t12.Description, t11.CoInsurance_Percent, t13.Transaction_Type, t14.patient_age, t14.Patient_Marital_Status,
t14.Patient_City, t14.Patient_State,
t14.patient_zip_code
FROM [PM].[Vouchers] t2 
left JOIN PM.Services t3 ON t2.Voucher_ID = t3.Voucher_ID  
left JOIN PM.vwGensvcInfo t4 ON t3.Service_ID = t4.Service_ID
left JOIN PM.ERA_Claims t5 ON t2.Voucher_ID = t5.Voucher_ID-- 2829929
left JOIN [PM].[Claim_Attachments] t6 ON t2.Voucher_ID = t6.Voucher_ID 
left JOIN PM.[Reimbursement_Detail] t7 ON t3.Service_ID = t7.Service_ID 
--left JOIN [PM].[ERA_Services] t8 ON t3.Service_ID = t8.Service_ID 
left JOIN PM.Patient_Policies t9 ON t9.Patient_Policy_ID = t5.Patient_Policy_ID 
left JOIN PM.Policies t10 ON t10.Policy_ID = t9.Policy_ID     
left JOIN PM.Insurance_Plans t11 ON t11.Insurance_Plan_ID = t10.Insurance_Plan_ID  
left JOIN [PM].[Reimbursement_Comments] t12 ON t7.Reimbursement_Detail_ID = t12.Reimbursement_Comment_ID 
left JOIN PM.vwGenSvcPmtInfo as t13  ON t4.Service_ID=t13.Service_ID
left JOIN PM.vwGenPatInfo as t14 ON t4.Patient_Number= t14.Patient_Number
WHERE (T4.Service_Fee >0) and t13.Transaction_Type='P' and (t13.Amount >= 0) and
            ((t4.Primary_Diagnosis_Code between 'E08' and 'E13') OR
            (t4.Primary_Diagnosis_Code between 'E66.0' and 'E66.99') OR 
            (t4.Primary_Diagnosis_Code between 'I10' and 'I16') OR 
            (t4.Primary_Diagnosis_Code between 'I25.00' and 'I25.99')) and 
            t2.Patient_ID = %s
ORDER BY t4.Service_Date_From DESC
"""
    cursor.execute(query, (patient_id,))
    db_response = cursor.fetchall()

    result = pd.DataFrame([list(elem) for elem in db_response])
    column_names = ['Voucher_ID', 'Voucher_Number', 'Claim_Number', 'Original_Billing_Date', 'Fees', 'Posted_Payments',
                    'Posted_Adjustments', 'Posted_Refunds', 'CoPayment_Amount', 'Patient_ID', 'Patient_Policy_ID',
                    'Is_Self_Pay', 'Service_ID', 'Service_Date_From', 'Original_Billing_Date',
                    'Date_Paid', 'Patient_Number', 'Original_Carrier_Name', 'Original_Carrier_Category_Abbr',
                    'Actual_Dr_Name', 'Place_of_Service_Descr', 'Procedure_Code', 'Service_Units', 'Service_Fee',
                    'Amount',
                    'Transaction_Code_Descr',
                    'Primary_Diagnosis_Code', 'Patient_Control_Number', 'Claim_Status',
                    'Total_Claim_Charge_Amount', 'Claim_Payment_Amount', 'Patient_Responsibility_Amount',
                    'Claim_Frequency_Code',
                    'Claim_DRG_Amount', 'Claim_PPS_Outlier_Amount', 'Claim_Stmt_Period_Start',
                    'Claim_Stmt_Period_End', 'Coverage_Expiration_Date', 'Claim_Received_Date',
                    'Allowed', 'Deductible', 'CoPayment', 'CoInsurance', 'EOB_Date', 'Reimbursement_Comment_ID',
                    'Reimbursement_Comment_Abbr', 'Pending',
                    'Denied', 'Coverage', 'Waive_CoPayment', 'Policy_ID', 'Verified_Date', 'Group_Name',
                    'Certificate_No', 'Comments',
                    'Effective_Date', 'Expiration_Date', 'Plan_Code',
                    'Description', 'Is_Capitated_Plan', 'Abbreviation',
                    'Description', 'CoInsurance_Percent', 'Transaction_Type', 'patient_age', 'Patient_Marital_Status',
                    'Patient_City', 'Patient_State',
                    'patient_zip_code']
    if result.empty:
        result = pd.DataFrame()
    else:
        result.columns = column_names
        result = result.loc[:, ~result.columns.duplicated()]

    return result


def assign_diagnosis_label(diagnosis_codes):
    """Function for identify whether ICD is belongs to Diabetic
    or Hypertension or Cardiovascular or obesity"""
    diabetic_range = (8.00, 13.00)
    hypertension_range = (10.00, 16.00)
    cardiovascular_range = (25.000, 25.999)
    obesity_range = (66.00, 66.99)
    diagnosis_label_list = []
    for code in diagnosis_codes:
        code = code.strip()
        code_number = float(code[1:])

        if code.startswith('E') and diabetic_range[0] <= code_number <= diabetic_range[1]:
            diagnosis_label_list.append('Diabetic')
        elif code.startswith('I') and hypertension_range[0] <= code_number <= hypertension_range[1]:
            diagnosis_label_list.append('Hypertension')
        elif code.startswith('I') and cardiovascular_range[0] <= code_number <= cardiovascular_range[1]:
            diagnosis_label_list.append('Cardiovascular')
        elif code.startswith('E') and obesity_range[0] <= code_number <= obesity_range[1]:
            diagnosis_label_list.append('Obesity')
        else:
            continue
    return diagnosis_label_list


def assign_score_to_colorband(score_list):
    """ Convert score to corresponding Band"""
    color_band = []
    for score in score_list:
        if score <= 0.5:
            color_band.append('Red')
        elif 0.5 < score <= 0.7:
            color_band.append('Orange')
        else:
            color_band.append('Green')
    return color_band


def top_denied_payer_update(payer):
    """Function for identify whether payer is top denied payer or not"""
    top_denied_payer_list = ClaimModelAppConfig.top_denied_payer
    if payer in top_denied_payer_list:
        return 1
    else:
        return 0


def top_denied_cpt_update(cpt_list):
    """Function for calculate the count of Top denied cpt in the combination"""
    top_denied_cpt_list = ClaimModelAppConfig.top_denied_cpt
    denied_cpt = 0
    for cpt in cpt_list:
        if cpt in top_denied_cpt_list:
            denied_cpt = denied_cpt + 1
        else:
            continue
    return denied_cpt


def get_marital_status(age):
    """Function to determine marital status based on age"""
    return 'Married' if age > 30 else 'Single'


def find_icd_and_score_for_cpt(cpt_code, recommended_code_list):
    """Given cpt mapping with ICD and Score"""
    for item in recommended_code_list:
        proc_code = item['Proc_code']
        if cpt_code in proc_code:
            icd_code = item['ICD']
            score = proc_code[cpt_code]
            return icd_code, score
    return None, None


class PredictClaim(APIView):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def post(self, request):
        start = time()
        top_accepted = {}
        top_denied = {}
        top_predict = {}
        request_data = request.data
        patient_id = request_data['Patient_ID']
        icd_cpt_score_list = request_data['recommended_code']
        icd_cpt_score_list = icd_cpt_score_list.replace("'", '"')
        icd_cpt_list = json.loads(icd_cpt_score_list)
        print("ICD_CPT_LIST: ", icd_cpt_list)
        proc_code_list = [list(entry['Proc_code'].keys()) for entry in icd_cpt_list]
        proc_code_list = [item for inner_list in proc_code_list for item in inner_list]
        # print(proc_code_list)   It gives all passed CPT Codes

        # combinations recommendation from CPT combinations #
        all_combinations = []
        for r in range(len(proc_code_list), 0, -1):
            combinations = itertools.combinations(proc_code_list, r)
            all_combinations.extend([list(combination) for combination in combinations])
        # print(all_combinations)    # It gives all possible combinations of CPT
        X = get_patient_details(self, patient_id)
        if X.shape[0] == 0:
            response = {
                "message": "Patient History is not available in DataBase",
                "status": "Success"
            }
            return Response(response)
        for cpt_recom in all_combinations:
            print("CPT_Recommendation :", cpt_recom)
            cpt_list = []
            icd_list = []
            score_list = []
            for cpt in cpt_recom:
                icd, score = find_icd_and_score_for_cpt(cpt, icd_cpt_list)
                cpt_list.append(cpt)
                icd_list.append(icd)
                score_list.append(score)
            print("CPT_list:", cpt_list)
            print("Icd_List:", icd_list)
            print("Score_List:", score_list)
            proc_code_list = list(set(cpt_list))
            icd_label = assign_diagnosis_label(icd_list)
            print("icd_label", icd_label)
            color_band = assign_score_to_colorband(score_list)
            predict_X = pd.DataFrame()
            feature_counts = {
                'Diabetic': icd_label.count('Diabetic'),
                'Hypertension': icd_label.count('Hypertension'),
                'Cardiovascular': icd_label.count('Cardiovascular'),
                'Obesity': icd_label.count('Obesity'),
                'Count_green': color_band.count('Green'),
                'Count_orange': color_band.count('Orange'),
                'Count_red': color_band.count('Red')
            }

            predict_X = predict_X.append(feature_counts, ignore_index=True)
            # Top Denied Cpt count #
            predict_X['count_top_denied_cpt'] = top_denied_cpt_update(proc_code_list)
            print("top_denied_CPT", predict_X['count_top_denied_cpt'])
            # Unique Cpt mapping #
            unique_cpt = ClaimModelAppConfig.unique_cpt
            unique_cpt = json.loads(unique_cpt)
            data_dict = {feature: [1 if feature in proc_code_list else 0] for feature in unique_cpt}
            print("data_dict_with_unique_CPT", data_dict)
            df_cpt = pd.DataFrame(data_dict, index=[0])
            predict_X = pd.concat([predict_X, df_cpt], axis=1)
            print(predict_X.head())
            # Adding features from patient History #
            pat_df = X
            predict_X['Service_Fee'] = int(pat_df["Service_Fee"].sum())
            predict_X['Patient_Responsibility_Amount'] = pat_df['Patient_Responsibility_Amount']
            predict_X['Is_Capitated_Plan'] = pat_df['Is_Capitated_Plan']
            predict_X['Original_Carrier_Name'] = pat_df['Original_Carrier_Name']
            predict_X['Patient_Marital_Status'] = pat_df['Patient_Marital_Status']
            predict_X['Place_of_Service_Descr'] = pat_df['Place_of_Service_Descr']
            predict_X['patient_age'] = pat_df['patient_age']
            predict_X['patient_zip_code'] = pat_df['patient_zip_code']
            predict_X['Patient_City'] = pat_df['Patient_City']
            predict_X['Patient_State'] = pat_df['Patient_State']
            # top Denied Payer #
            predict_X['top_denied_payer'] = predict_X['Original_Carrier_Name'].apply(top_denied_payer_update)
            # Handling Missing Values #
            predict_X['Patient_Marital_Status'] = predict_X.apply(
                lambda row: get_marital_status(row['patient_age']) if pd.isnull(row['Patient_Marital_Status']) else row[
                    'Patient_Marital_Status'], axis=1)
            predict_X['Is_Capitated_Plan'] = predict_X['Is_Capitated_Plan'].fillna(True)
            predict_X['Patient_Responsibility_Amount'] = predict_X['Patient_Responsibility_Amount'].fillna(0)
            predict_X['Patient_Responsibility_Amount'] = predict_X['Patient_Responsibility_Amount'].astype(int)
            predict_X['Is_Capitated_Plan'] = predict_X['Is_Capitated_Plan'].astype(int)
            # Label Encoding #
            payer_mapping_dict = ClaimModelAppConfig.payer_mapping['payer_mapping']
            predict_X = predict_X.replace({"Original_Carrier_Name": payer_mapping_dict})
            for col in ClaimModelAppConfig.categorical_columns:
                # Get the label encoder for the current column
                le = ClaimModelAppConfig.label_encoders[col]
                # Apply label encoding to the NumPy array
                try:
                    new_data_encoded = le.transform(predict_X[col])

                    # Assign the transformed values back to predict_X DataFrame
                    predict_X[col] = new_data_encoded
                except ValueError:
                    # Some variables having trailing spaces in train set but not at prediction time.
                    col_classes = predict_X[col].unique().tolist()
                    le_classes = le.classes_.tolist()
                    missing_classes = [lbl for lbl in col_classes if lbl not in le_classes]
                    missing_indx_dict = {}
                    le_dict = {i: idx for idx, i in enumerate(le_classes)}

                    classes_to_replace = []
                    for missing_cls in missing_classes:

                        missing_cls_trimmed = missing_cls.strip()
                        indx = None
                        if missing_cls_trimmed in le_classes:
                            indx = le_classes.index(missing_cls_trimmed)
                        else:
                            le_classes_trimmed = [i.strip() for i in le_classes]
                            if missing_cls_trimmed in le_classes_trimmed:
                                indx = le_classes_trimmed.index(missing_cls_trimmed)
                        if indx:
                            missing_indx_dict[missing_cls] = indx
                        elif not indx:
                            classes_to_replace.append(missing_cls)

                    le_dict.update(missing_indx_dict)

                    if classes_to_replace:
                        self.logger.warning("New payer names found, which are not present at training time")
                        self.logger.info("Missing classes in payer column are:")
                        for c, missing_payer in enumerate(classes_to_replace):
                            self.logger.info("---->{}: {}".format(c + 1, missing_payer))
                        frequent_payer_index = le_classes.index(ClaimModelAppConfig.most_frequent_payer)
                        replace_missing_dict = dict(
                            zip(classes_to_replace, [frequent_payer_index] * len(classes_to_replace)))
                        le_dict.update(replace_missing_dict)

                    predict_X[col] = [le_dict[label] for label in predict_X[col]]

            # Preprocessing Completed #
            predict_X = predict_X[ClaimModelAppConfig.xgb_model_fit_feature_order]
            print("Lenth of actual data columns", len(predict_X))
            # Model Predict Probability Score
            prediction_probability = ClaimModelAppConfig.xgb_model.predict_proba(predict_X)
            # Predict Claim Status
            predict_result = ClaimModelAppConfig.xgb_model.predict(predict_X)
            top_accepted[str(cpt_recom)] = prediction_probability[0][0]
            top_denied[str(cpt_recom)] = prediction_probability[0][1]
            top_predict[str(cpt_recom)] = predict_result[0]

            if len(top_accepted) < 5:
                n = len(top_accepted)
            else:
                n = 5
        # top_accepted_rounded = {key: round(value, 7) for key, value in top_accepted.items()}
        top_five_pairs_accepted = heapq.nlargest(n, top_accepted.items(), key=lambda item: item[1])
        # top_denied_rounded = {key: round(value, 7) for key, value in top_denied.items()}
        top_five_pairs_denied = heapq.nlargest(n, top_denied.items(), key=lambda item: item[1])
        top_five_pairs_accepted = dict(top_five_pairs_accepted)
        top_five_pairs_denied = dict(top_five_pairs_denied)
        top_accepted_cpt = list(top_five_pairs_accepted.keys())
        top_denied_cpt = list(top_five_pairs_denied.keys())
        end = time()
        response = {
            "message": "Prediction Engine Service completed successfully.",
            "status": "Success",
            "statusCode": 200,
            "respTime": round(end - start, 3),
            "Patient_ID": str(patient_id),
            "Top_Five_Claim_Accepted_Cpt_Combination": top_accepted_cpt,
            "Top_Five_Claim_Denied_Cpt_Combination": top_denied_cpt
        }

        return Response(response)
