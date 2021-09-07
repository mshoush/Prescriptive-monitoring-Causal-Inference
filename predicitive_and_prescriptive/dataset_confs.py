import os

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
treatment_col = {}
pos_treatment = {}
neg_treatment = {}

pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}

logs_dir = "/home/centos/phd/code/github/predictive_and_prescriptive"
#logs_dir = "/home/mshoush/ut_cs_phd/phd/code/gitHub/predictive_and_prescriptive"

#### BPIC2017 settings ####

bpic2017_dict = {"prepared_bpic2017": "prepared_bpic2017.pkl"}

for dataset, fname in bpic2017_dict.items():
    filename[dataset] = os.path.join(logs_dir, fname)
    # min cols
    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = 'org:resource'
    timestamp_col[dataset] = 'time:timestamp'
    # label/outcome col
    label_col[dataset] = "label"
    neg_label[dataset] = "regular"  # negative outcome that we don't need to predict
    pos_label[dataset] = "deviant"  # positive outcome that will be predicted
    # treatment col
    treatment_col[dataset] = 'treatment'
    pos_treatment[dataset] = "treat"  # do treatment
    neg_treatment[dataset] = "noTreat"  # do not treat

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", 'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition', "Accepted", "Selected"]
    static_cat_cols[dataset] = ['ApplicationType', 'LoanGoal']
    dynamic_num_cols[dataset] = ['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', 'CreditScore', "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour","open_cases"]
    static_num_cols[dataset] = ['RequestedAmount', "NumberOfOffers"]
