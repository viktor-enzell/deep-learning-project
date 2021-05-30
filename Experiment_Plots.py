import numpy as np
import matplotlib.pyplot as plt
from csv import reader
import os
import pandas as pd

# current_path = os.path.dirname(os.path.abspath(__file__))
# file_name = "Transformer.csv"
# file_path = os.path.join(current_path, file_name)

# opened_file = open(file_path)

# read_file = reader(opened_file)

# data = list(read_file)

"LOSS DATA THAT IS GENERATED FROM MODELS DURING TRAINING"
RNN_main = np.array([43.13427665,42.0830938,41.54901794,41.2136166,40.9907408,40.8327906,40.71711881,40.62676248,40.55060979,40.48212099,\
                     40.41909235,40.36163307,40.30960764,40.26214833,40.21906747,40.18034837,40.14488476,40.1127368,40.0831316,40.05587862])
    
RNN_1 = np.array([56.98566725,52.02870656,50.63935241,49.33925086,48.94218468,48.34068795,47.66049431,47.71939079,47.23248597,47.34786994])
RNN_2 = np.array([52.85264928,48.62357473,47.16314196,46.24294839,46.22026523,45.71308849,46.20600371,45.21322371,45.28596689,44.76896209])

RNN_3 = np.array([53.97977488,51.48582951,49.94818372,49.23057135,48.57451551,47.56241545,47.11012676,46.55301375,46.0722682,45.70135865])
RNN_4 = np.array([100.3228469,102.7633949,103.904359,104.603011,105.088319,None,None,None,None,None])

RNN_5 = np.array([48.45584161,46.91872891,46.18728634,45.72541231,45.37609633,45.11128271,44.89010695,44.70150371,44.54366719,44.40285522,\
                  44.36833044,44.20922302,44.16740306,44.04513417,43.94642008,43.84703565,43.79670021,43.7143153,43.63740825,43.56388185])
RNN_6 = np.array([84.14669155,84.35909136,84.47409302,84.56811667,84.58042129,84.62221564,84.66069128,84.69109066,84.71162877,84.72924694])


LSTM_main = np.array([1.466155245,1.29587622,1.240812144,1.205953035,1.178585668,1.159236337,1.141239026,1.127767054,1.114618113,\
                      1.102477134,1.091331698,1.080750223,1.071427198,1.06146004,1.052281501,1.043599138,1.036339305,1.028250988,\
                          1.020402992,1.014048412,1.005536801,0.998350898,0.9913817731,0.9849554793,0.9785186872])
    
LSTM_1 = np.array([2.112427546,1.52554022,1.414575537,1.34940554,1.301537854,1.263680105,1.232102354,1.202933113,1.17750786,\
                   1.152592039,1.133014958,1.109199992,1.088668967,1.068506418,1.054994974,1.033125479,1.013806425,0.9969144951,\
                       0.9758696572,0.956933868,0.9382156298,0.9191584323,0.9003652324,0.8835232808,0.866588711])
LSTM_2 = np.array([1.549586971,1.406377562,1.356607799,1.313889417,1.278511475,1.247654845,1.219689187,1.197506425,\
                   1.178708722,1.157570424,1.141747648,1.125462931,1.109950829,1.094761321,1.082942112,1.068468795,\
                       1.055871644,1.044264078,1.031084662,1.018666347,1.00766663,0.9973206168,0.9858636781,0.9764270261,0.9661184005])


LSTM_3 = np.array([3.544949261,3.435921479,3.516639072,3.502750988,3.481838053,3.461304189,3.447126967,3.429834711,\
                   3.410040876,3.400803278,3.3861653,3.373181018,3.360317642,3.348035093,3.336678515,3.325680733,3.313791027,\
                       3.305653771,3.29504997,3.283193828,3.276435459,3.268405619,3.261738835,3.251844971,3.245243667])
LSTM_4 = np.array([1.513650623,1.184587089,1.088111657,1.02629296,0.976291406,0.9315146347,0.8932813185,0.8547537884,0.8172027215,\
                   0.7850433683,0.752191588,0.7222095729,0.6936783011,0.6640191654,0.6345654914,0.6088458627,0.5827985954,0.5576742625,\
                       0.5340190103,0.5131476222,0.4908222006,0.4742285706,0.4540083569,0.4364669094,0.4183817188])

    
LSTM_5 = np.array([1.646635052,1.444512225,1.401442899,1.377845014,1.361027593,1.344520191,1.328226974,1.31810616,1.30720595,1.298937788,\
                   1.288918143,1.280675879,1.27339197,1.266440375,1.262671457,1.255071656,1.251448394,1.245332859,1.240183506,1.232896013,\
                       1.227558113,1.223731218,1.218192758,1.214181479,1.210878334])
LSTM_6 = np.array([1.512461975,1.345240611,1.30607368,1.281229319,1.262658747,1.245567864,1.230719218,1.21771695,1.204836156,1.19442531,\
                   1.184147516,1.173910689,1.165652787,1.156499182,1.148643154,1.141355925,1.134708153,1.127217614,1.121698198,1.115123604,\
                       1.110226454,1.104305375,1.098381749,1.092933086,1.088280431])
    
transformer_main = np.array([5.45,4.79,4.52,4.32,4.13,3.96,3.82,3.66,3.53,3.38,3.25,3.11,3.01,2.89,2.79,2.68,2.57,\
                             2.48,2.39,2.31,2.23,2.14,2.07,2.01,1.94,1.88,1.82,1.77,1.72,1.67,1.63,1.59,1.56,1.51,1.47,\
                                 1.45,1.42,1.38,1.36,1.33,1.31,1.28,1.26,1.24,1.22,1.2,1.19,1.17,1.15,1.14,1.12,1.11,1.1,1.09,1.08,1.07,\
                                     1.06,1.05,1.04,1.03,1.02,1.02,1.01,1,1,0.99,0.99,0.98,0.97,0.97,0.97,0.96,0.96,0.95,0.95,0.95,0.94,0.94,0.94,\
                                     0.94,0.93,0.93,0.93,0.93,0.93,0.92,0.92,0.92,0.92,0.92,0.92,0.91,0.91,0.91,0.91,0.91,0.91,0.91,0.91,0.91])
    
transformer_1 = np.array([8.05,5.56,4.7,4.33,4.02,3.65,3.34,3.07,2.79,2.54,2.31,2.14,1.94,1.76,1.6,1.46,1.34,1.23,1.13,1.02])
transformer_2 = np.array([6.09,5.11,4.66,4.35,4.08,3.84,3.66,3.44,3.25,3.09,2.91,2.75,2.6,2.46,2.33,2.21,2.08,1.98,1.88,1.77])

transformer_3 = np.array([5.92,5.58,5.34,5.18,5.03,4.93,4.84,4.76,4.7,4.63,4.57,4.52,4.47,4.43,4.39,4.36,4.32,4.29,4.26,4.23])
transformer_4 = np.array([9.8,7.31,7.28,6.65,6.32,6.29,6.14,6.12,6.13,6.07,5.92,5.87,5.92,5.71,5.46,5.38,5.26,5.08,5.02,4.89])

transformer_5 = np.array([5.67,5.01,4.73,4.54,4.4,4.27,4.14,4.06,3.96,3.88,3.8,3.75,3.68,3.62,3.56,3.5,3.46,3.41,3.37,3.32])
transformer_6 = np.array([5.47,4.85,4.59,4.37,4.23,4.09,3.94,3.83,3.7,3.59,3.49,3.39,3.3,3.21,3.14,3.04,2.97,2.89,2.83,2.76])


"LIST OF LOSS VALUES FOR EACH MODEL OF EACH NETWORK"


RNN_models = [RNN_main, RNN_1, RNN_2, RNN_3, RNN_4, RNN_5, RNN_6]

LSTM_models = [LSTM_main, LSTM_1, LSTM_2, LSTM_3, LSTM_4, LSTM_5, LSTM_6]

transformer_models = [transformer_main, transformer_1, transformer_2, transformer_3, transformer_4, transformer_5, transformer_6]


"CREATING LISTS THAT CONTAINS LOSS DROP IN TERMS OF PERCENTAGES FOR 10 EPOCHS FOR EACH MODEL IN EACH NETWORK"
RNN_loss_drop = []
LSTM_loss_drop = []
transformer_loss_drop = []

for i in range(len(RNN_models)):
    epoch = 9 #9+1
    RNN_model = RNN_models[i]
    LSTM_model = LSTM_models[i]
    transformer_model = transformer_models[i]
    #Since model-4 of RNN has only 5 loss values, we calculate only 5 Epochs loss drop for this model
    if i == 4:
        RNN_loss = (RNN_model[0]-RNN_model[4]) / RNN_model[0] * 100
    else:
        RNN_loss = (RNN_model[0]-RNN_model[epoch]) / RNN_model[0] * 100
    
    LSTM_loss = (LSTM_model[0]-LSTM_model[epoch]) / LSTM_model[0] * 100
    transformer_loss = (transformer_model[0]-transformer_model[epoch]) / transformer_model[0] * 100
    RNN_loss_drop.append(RNN_loss)
    LSTM_loss_drop.append(LSTM_loss)
    transformer_loss_drop.append(transformer_loss)


" TABLE FOR ALL MODELS FIGURE"
# columns = ["Network Type", "Model", "Amount of Training Data (%)", "Learning Rate", "# of Nodes per layer"]
# columns = [{"Network Type":1, "Model":2, "Amount of Training Data (%)":3, "Learning Rate":4, "# of Nodes per layer":5}]
# cell_values = [["RNN", "Main Model", "100%", "0.1", "50"],
#                 ["RNN", "Model-1", "30%", "0.1", "100"],
#                 ["RNN", "Model-2", "60%", "0.1", "100"],
#                 ["RNN", "Model-3", "100%", "0.01", "100"],
#                 ["RNN", "Model-4", "100%", "1", "100"],
#                 ["RNN", "Model-5", "100%", "0.1", "100"],
#                 ["RNN", "Model-6", "100%", "0.1", "200"],
#                 ["LSTM", "Main Model", "100%", "0.002", "512"],
#                 ["LSTM", "Model-1", "30%", "0.002", "512"],
#                 ["LSTM", "Model-2", "60%", "0.002", "512"],
#                 ["LSTM", "Model-3", "100%", "0.02", "512"],
#                 ["LSTM", "Model-4", "100%", "0.0002", "512"],
#                 ["LSTM", "Model-5", "100%", "0.002", "128"],
#                 ["LSTM", "Model-6", "100%", "0.002", "256"],
#                 ["Transformer", "Main Model", "100%", "5", "512"],
#                 ["Transformer", "Model-1", "30%", "5", "512"],
#                 ["Transformer", "Model-2", "60%", "5", "512"],
#                 ["Transformer", "Model-3", "100%", "0.5", "512"],
#                 ["Transformer", "Model-4", "100%", "10", "512"],
#                 ["Transformer", "Model-5", "100%", "5", "128"],
#                 ["Transformer", "Model-6", "100%", "5", "256"]]
# dff_columns = pd.DataFrame(columns)
# dff_values = pd.DataFrame(cell_values)
   
# fig, ax = plt.subplots() 
# ax.set_axis_off() 
# table = ax.table( 
#     cellText = dff_values.values,  
#     # rowLabels = val2,  
#     colLabels = dff_columns.columns, 
#     # rowColours =["palegreen"] * 10,  
#     colColours =["yellow"] * 5, 
#     cellLoc ='center',  
#     loc ='upper left')         
   
# ax.set_title('Models for The Experiments', 
#               fontweight ="bold", fontsize = 25) 
# table.auto_set_font_size(False)
# table.set_fontsize(18)
# table.auto_set_column_width(col=list(range(len(dff_values.columns))))
# table[(2,2)].set_facecolor("lightcoral")
# table[(3,2)].set_facecolor("lightcoral")
# table[(9,2)].set_facecolor("lightcoral")
# table[(10,2)].set_facecolor("lightcoral")
# table[(16,2)].set_facecolor("lightcoral")
# table[(17,2)].set_facecolor("lightcoral")


# table[(4, 3)].set_facecolor("lightcoral")
# table[(5, 3)].set_facecolor("lightcoral")
# table[(11, 3)].set_facecolor("lightcoral")
# table[(12, 3)].set_facecolor("lightcoral")
# table[(18, 3)].set_facecolor("lightcoral")
# table[(19, 3)].set_facecolor("lightcoral")


# table[(6, 4)].set_facecolor("lightcoral")
# table[(7, 4)].set_facecolor("lightcoral")
# table[(13, 4)].set_facecolor("lightcoral")
# table[(14, 4)].set_facecolor("lightcoral")
# table[(20, 4)].set_facecolor("lightcoral")
# table[(21, 4)].set_facecolor("lightcoral")



"TABLE FOR MODELS FOR TRAINING DATA FIGURE"


# columns = ["Network Type", "Model", "Amount of Training Data (%)", "Learning Rate", "# of Nodes per layer"]
# columns = [{"Network Type":1, "Model":2, "Amount of Training Data (%)":3, "Learning Rate":4, "# of Nodes per layer":5}]
# cell_values = [["RNN", "Main Model", "100%", "0.1", "50*"],
#                ["RNN", "Model-1", "30%", "0.1", "100"],
#                ["RNN", "Model-2", "60%", "0.1", "100"],

#                ["LSTM", "Main Model", "100%", "0.002", "512"],
#                ["LSTM", "Model-1", "30%", "0.002", "512"],
#                ["LSTM", "Model-2", "60%", "0.002", "512"],

#                ["Transformer", "Main Model", "100%", "5", "512"],
#                ["Transformer", "Model-1", "30%", "5", "512"],
#                ["Transformer", "Model-2", "60%", "5", "512"]]


# dff_columns = pd.DataFrame(columns)
# dff_values = pd.DataFrame(cell_values)
   
# fig, ax = plt.subplots() 
# ax.set_axis_off() 
# table = ax.table( 
#     cellText = dff_values.values,  
#     # rowLabels = val2,  
#     colLabels = dff_columns.columns, 
#     # rowColours =["palegreen"] * 10,  
#     colColours =["yellow"] * 5, 
#     cellLoc ='center',  
#     loc ='upper left')         
   
# ax.set_title('Models for Training Data Experiment', 
#              fontweight ="bold", fontsize = 25) 
# table.auto_set_font_size(False)
# table.set_fontsize(18)
# table.auto_set_column_width(col=list(range(len(dff_values.columns))))

# table[(1, 2)].set_facecolor("lightcoral")
# table[(2, 2)].set_facecolor("lightcoral")
# table[(3, 2)].set_facecolor("lightcoral")
# table[(4, 2)].set_facecolor("lightcoral")
# table[(5, 2)].set_facecolor("lightcoral")
# table[(6, 2)].set_facecolor("lightcoral")
# table[(7, 2)].set_facecolor("lightcoral")
# table[(8, 2)].set_facecolor("lightcoral")
# table[(9, 2)].set_facecolor("lightcoral")




"TABLE MODELS FOR lEARNING RATE FIGURE"


# columns = ["Network Type", "Model", "Amount of Training Data (%)", "Learning Rate", "# of Nodes per layer"]
# columns = [{"Network Type":1, "Model":2, "Amount of Training Data (%)":3, "Learning Rate":4, "# of Nodes per layer":5}]
# cell_values = [["RNN", "Main Model", "100%", "0.1", "50*"],

#                 ["RNN", "Model-3", "100%", "0.01", "100"],
#                 ["RNN", "Model-4", "100%", "1", "100"],
#                 ["LSTM", "Main Model", "100%", "0.002", "512"],

#                 ["LSTM", "Model-3", "100%", "0.02", "512"],
#                 ["LSTM", "Model-4", "100%", "0.0002", "512"],

#                 ["Transformer", "Main Model", "100%", "5", "512"],

#                 ["Transformer", "Model-3", "100%", "0.5", "512"],
#                 ["Transformer", "Model-4", "100%", "10", "512"]]
# dff_columns = pd.DataFrame(columns)
# dff_values = pd.DataFrame(cell_values)
   
# fig, ax = plt.subplots() 
# ax.set_axis_off() 
# table = ax.table( 
#     cellText = dff_values.values,  
#     # rowLabels = val2,  
#     colLabels = dff_columns.columns, 
#     # rowColours =["palegreen"] * 10,  
#     colColours =["yellow"] * 5, 
#     cellLoc ='center',  
#     loc ='upper left')         
   
# ax.set_title('Models for Learning Rate Experiment', 
#               fontweight ="bold", fontsize = 25) 
# table.auto_set_font_size(False)
# table.set_fontsize(18)
# table.auto_set_column_width(col=list(range(len(dff_values.columns))))

# table[(1, 3)].set_facecolor("lightcoral")
# table[(2, 3)].set_facecolor("lightcoral")
# table[(3, 3)].set_facecolor("lightcoral")
# table[(4, 3)].set_facecolor("lightcoral")
# table[(5, 3)].set_facecolor("lightcoral")
# table[(6, 3)].set_facecolor("lightcoral")
# table[(7, 3)].set_facecolor("lightcoral")
# table[(8, 3)].set_facecolor("lightcoral")
# table[(9, 3)].set_facecolor("lightcoral")

"TABLE FOR MODELS FOR NUMBER OF NODES EACH LAYER"

# columns = ["Network Type", "Model", "Amount of Training Data (%)", "Learning Rate", "# of Nodes per layer"]
# columns = [{"Network Type":1, "Model":2, "Amount of Training Data (%)":3, "Learning Rate":4, "# of Nodes per layer":5}]
# cell_values = [["RNN", "Main Model", "100%", "0.1", "50"],

#                 ["RNN", "Model-5", "100%", "0.1", "100"],
#                 ["RNN", "Model-6", "100%", "0.1", "200"],
#                 ["LSTM", "Main Model", "100%", "0.002", "512"],

#                 ["LSTM", "Model-5", "100%", "0.002", "128"],
#                 ["LSTM", "Model-6", "100%", "0.002", "256"],
                
#                 ["Transformer", "Main Model", "100%", "5", "512"],

#                 ["Transformer", "Model-5", "100%", "5", "128"],
#                 ["Transformer", "Model-6", "100%", "5", "256"],]
# dff_columns = pd.DataFrame(columns)
# dff_values = pd.DataFrame(cell_values)
   
# fig, ax = plt.subplots() 
# ax.set_axis_off() 
# table = ax.table( 
#     cellText = dff_values.values,  
#     # rowLabels = val2,  
#     colLabels = dff_columns.columns, 
#     # rowColours =["palegreen"] * 10,  
#     colColours =["yellow"] * 5, 
#     cellLoc ='center',  
#     loc ='upper left')         
   
# ax.set_title('Models for Number of Nodes Experiment', 
#               fontweight ="bold", fontsize = 25) 
# table.auto_set_font_size(False)
# table.set_fontsize(18)
# table.auto_set_column_width(col=list(range(len(dff_values.columns))))

# table[(1, 4)].set_facecolor("lightcoral")
# table[(2, 4)].set_facecolor("lightcoral")
# table[(3, 4)].set_facecolor("lightcoral")
# table[(4, 4)].set_facecolor("lightcoral")
# table[(5, 4)].set_facecolor("lightcoral")
# table[(6, 4)].set_facecolor("lightcoral")
# table[(7, 4)].set_facecolor("lightcoral")
# table[(8, 4)].set_facecolor("lightcoral")
# table[(9, 4)].set_facecolor("lightcoral")


"TABLE FOR MODELS FOR MAINS"

# columns = ["Network Type", "Model", "Amount of Training Data (%)", "Learning Rate", "# of Nodes per layer"]
# columns = [{"Network Type":1, "Model":2, "Amount of Training Data (%)":3, "Learning Rate":4, "# of Nodes per layer":5}]
# cell_values = [["RNN", "Main Model", "100%", "0.1", "50"],

#                 ["LSTM", "Main Model", "100%", "0.002", "512"],

#                 ["Transformer", "Main Model", "100%", "5", "512"]]

# dff_columns = pd.DataFrame(columns)
# dff_values = pd.DataFrame(cell_values)
   
# fig, ax = plt.subplots() 
# ax.set_axis_off() 
# table = ax.table( 
#     cellText = dff_values.values,  
#     # rowLabels = val2,  
#     colLabels = dff_columns.columns, 
#     # rowColours =["palegreen"] * 10,  
#     colColours =["yellow"] * 5, 
#     cellLoc ='center',  
#     loc ='upper left')         
   
# ax.set_title('Models for Network Comparison', 
#               fontweight ="bold", fontsize = 25) 
# table.auto_set_font_size(False)
# table.set_fontsize(18)
# table.auto_set_column_width(col=list(range(len(dff_values.columns))))

# table[(1, 0)].set_facecolor("lightcoral")
# table[(2, 0)].set_facecolor("lightcoral")
# table[(3, 0)].set_facecolor("lightcoral")


"TRAINING DATA EXPERIMENT PLOT COMPARISON WITH LOSS DROP PERCENTAGE"

# f1, axarr1 = plt.subplots(2, 2)
# axarr1 = axarr1.reshape(2,2)
# f1.suptitle('Training Data Loss Comparisons',fontweight ="bold", fontsize=25)

# x_axis1 = list(range(11))
# x_axis1 = np.array(x_axis1[1:])

# axarr1[0, 0].plot(x_axis1, RNN_1)
# axarr1[0, 0].plot(x_axis1, RNN_2)
# axarr1[0, 0].plot(x_axis1, RNN_main[:10])
# axarr1[0, 0].set_xticks(x_axis1)
# axarr1[0, 0].set_title("RNN Network",fontweight ="bold", fontsize=15)
# axarr1[0, 0].set(xlabel = "epochs", ylabel = "loss")
# axarr1[0, 0].legend(["Trained Data: 30% (Model-1)", "Trained Data: 60% (Model-2)","Trained Data: 100% (Main Model)"])
                    
# axarr1[0, 1].plot(x_axis1, LSTM_1[:10])
# axarr1[0, 1].plot(x_axis1, LSTM_2[:10])
# axarr1[0, 1].plot(x_axis1, LSTM_main[:10])
# axarr1[0, 1].set_xticks(x_axis1)
# axarr1[0, 1].set_title("LSTM Network",fontweight ="bold", fontsize=15)
# axarr1[0, 1].set(xlabel = "epochs", ylabel = "loss")
# axarr1[0, 1].legend(["Trained Data: 30% (Model-1)", "Trained Data: 60% (Model-2)","Trained Data: 100% (Main Model)"])

# axarr1[1, 0].plot(x_axis1, transformer_1[:10])
# axarr1[1, 0].plot(x_axis1, transformer_2[:10])
# axarr1[1, 0].plot(x_axis1, transformer_main[:10])
# axarr1[1, 0].set_xticks(x_axis1)
# axarr1[1, 0].set_title("Transformer Network",fontweight ="bold", fontsize=15)
# axarr1[1, 0].set(xlabel = "epochs", ylabel = "loss")
# axarr1[1, 0].legend(["Trained Data: 30% (Model-1)", "Trained Data: 60% (Model-2)","Trained Data: 100% (Main Model)"])

# columns = [{"Network Type":1, "Model":2, "Loss drop in first 10 epochs (%)":3}]
# cell_values = [["RNN", "Main Model (Trained Data: 100%)", str(RNN_loss_drop[0]) +" %"],
#                 ["RNN", "Model-1 (Trained Data: 30%)", str(RNN_loss_drop[1]) +" %"],
#                 ["RNN", "Model-2 (Trained Data: 60%)", str(RNN_loss_drop[2]) +" %"],

#                 ["LSTM", "Main Model (Trained Data: 100%)", str(LSTM_loss_drop[0]) +" %"],
#                 ["LSTM", "Model-1 (Trained Data: 30%)", str(LSTM_loss_drop[1]) +" %"],
#                 ["LSTM", "Model-2 (Trained Data: 60%)", str(LSTM_loss_drop[2]) +" %"],

#                 ["Transformer", "Main Model (Trained Data: 100%)", str(transformer_loss_drop[0]) +" %"],
#                 ["Transformer", "Model-1 (Trained Data: 30%)", str(transformer_loss_drop[1]) +" %"],
#                 ["Transformer", "Model-2 (Trained Data: 60%)", str(transformer_loss_drop[2]) +" %"]]

# dff_columns = pd.DataFrame(columns)
# dff_values = pd.DataFrame(cell_values)

   

# axarr1[1,1].set_axis_off() 
# table = axarr1[1,1].table( 
#     cellText = dff_values.values,  
#     # rowLabels = val2,  
#     colLabels = dff_columns.columns, 
#     # rowColours =["palegreen"] * 10,  
#     colColours =["yellow"] * 3, 
#     cellLoc ='center',  
#     loc ='upper left')         
   
# axarr1[1,1].set_title('Loss Drop Comparison (%)', 
#               fontweight ="bold", fontsize = 15) 
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.auto_set_column_width(col=list(range(len(dff_values.columns))))

# table[(1, 2)].set_facecolor("palegreen")
# table[(2, 2)].set_facecolor("palegreen")
# table[(3, 2)].set_facecolor("palegreen")
# table[(4, 2)].set_facecolor("palegreen")
# table[(5, 2)].set_facecolor("palegreen")
# table[(6, 2)].set_facecolor("palegreen")
# table[(7, 2)].set_facecolor("palegreen")
# table[(8, 2)].set_facecolor("palegreen")
# table[(9, 2)].set_facecolor("palegreen")



"LEARNING RATE EXPERIMENT PLOT COMPARISON WITH LOSS DROP PERCENTAGE"


# f1, axarr1 = plt.subplots(2, 2)
# axarr1 = axarr1.reshape(2,2)
# f1.suptitle('Learning Rate Loss Comparisons',fontweight ="bold", fontsize=25)

# x_axis1 = list(range(11))
# x_axis1 = np.array(x_axis1[1:])

# axarr1[0, 0].plot(x_axis1, RNN_3)
# axarr1[0, 0].plot(x_axis1, RNN_4)
# axarr1[0, 0].plot(x_axis1, RNN_main[:10])
# axarr1[0, 0].set_xticks(x_axis1)
# axarr1[0, 0].set_title("RNN Network",fontweight ="bold", fontsize=15)
# axarr1[0, 0].set(xlabel = "epochs", ylabel = "loss")
# axarr1[0, 0].legend(["Learning Rate: 0.01 (Model-3)", "Learning Rate: 1 (Model-4)","Learning Rate: 0.1 (Main Model)"])
                    
# axarr1[0, 1].plot(x_axis1, LSTM_3[:10])
# axarr1[0, 1].plot(x_axis1, LSTM_4[:10])
# axarr1[0, 1].plot(x_axis1, LSTM_main[:10])
# axarr1[0, 1].set_xticks(x_axis1)
# axarr1[0, 1].set_title("LSTM Network",fontweight ="bold", fontsize=15)
# axarr1[0, 1].set(xlabel = "epochs", ylabel = "loss")
# axarr1[0, 1].legend(["Learning Rate: 0.02 (Model-3)", "Learning Rate: 0.0002 (Model-4)","Learning Rate: 0.002 (Main Model)"])

# axarr1[1, 0].plot(x_axis1, transformer_3[:10])
# axarr1[1, 0].plot(x_axis1, transformer_4[:10])
# axarr1[1, 0].plot(x_axis1, transformer_main[:10])
# axarr1[1, 0].set_xticks(x_axis1)
# axarr1[1, 0].set_title("Transformer Network",fontweight ="bold", fontsize=15)
# axarr1[1, 0].set(xlabel = "epochs", ylabel = "loss")
# axarr1[1, 0].legend(["Learning Rate: 0.5 (Model-3)", "Learning Rate: 10 (Model-4)","Learning Rate: 5 (Main Model)"])

# columns = [{"Network Type":1, "Model":2, "Loss drop in first 10 epochs (%)":3}]
# cell_values = [["RNN", "Main Model (Learning Rate: 0.1)", str(RNN_loss_drop[0]) +" %"],
#                 ["RNN", "Model-3 (Learning Rate: 0.01)", str(RNN_loss_drop[3]) +" %"],
#                 ["RNN", "Model-4 (Learning Rate: 1)", str(RNN_loss_drop[4]) +" %"],

#                 ["LSTM", "Main Model (Learning Rate: 0.002)", str(LSTM_loss_drop[0]) +" %"],
#                 ["LSTM", "Model-3 (Learning Rate: 0.02)", str(LSTM_loss_drop[3]) +" %"],
#                 ["LSTM", "Model-4 (Learning Rate: 0.0002)", str(LSTM_loss_drop[4]) +" %"],

#                 ["Transformer", "Main Model (Learning Rate: 5)", str(transformer_loss_drop[0]) +" %"],
#                 ["Transformer", "Model-3 (Learning Rate: 0.5)", str(transformer_loss_drop[3]) +" %"],
#                 ["Transformer", "Model-4 (Learning Rate: 10)", str(transformer_loss_drop[4]) +" %"]]

# dff_columns = pd.DataFrame(columns)
# dff_values = pd.DataFrame(cell_values)

   

# axarr1[1,1].set_axis_off() 
# table = axarr1[1,1].table( 
#     cellText = dff_values.values,  
#     # rowLabels = val2,  
#     colLabels = dff_columns.columns, 
#     # rowColours =["palegreen"] * 10,  
#     colColours =["yellow"] * 3, 
#     cellLoc ='center',  
#     loc ='upper left')         
   
# axarr1[1,1].set_title('Loss Drop Comparison (%)', 
#               fontweight ="bold", fontsize = 15) 
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.auto_set_column_width(col=list(range(len(dff_values.columns))))

# table[(1, 2)].set_facecolor("palegreen")
# table[(2, 2)].set_facecolor("palegreen")
# table[(3, 2)].set_facecolor("palegreen")
# table[(4, 2)].set_facecolor("palegreen")
# table[(5, 2)].set_facecolor("palegreen")
# table[(6, 2)].set_facecolor("palegreen")
# table[(7, 2)].set_facecolor("palegreen")
# table[(8, 2)].set_facecolor("palegreen")
# table[(9, 2)].set_facecolor("palegreen")

"NUMBER OF NODES EXPERIMENT PLOT COMPARISON WITH LOSS DROP PERCENTAGE"


# f1, axarr1 = plt.subplots(2, 2)
# axarr1 = axarr1.reshape(2,2)
# f1.suptitle('Number of Nodes Loss Comparisons',fontweight ="bold", fontsize=25)

# x_axis1 = list(range(11))
# x_axis1 = np.array(x_axis1[1:])

# axarr1[0, 0].plot(x_axis1, RNN_5[:10])
# axarr1[0, 0].plot(x_axis1, RNN_6[:10])
# axarr1[0, 0].plot(x_axis1, RNN_main[:10])
# axarr1[0, 0].set_xticks(x_axis1)
# axarr1[0, 0].set_title("RNN Network",fontweight ="bold", fontsize=15)
# axarr1[0, 0].set(xlabel = "epochs", ylabel = "loss")
# axarr1[0, 0].legend(["Number of Nodes: 100 (Model-5)", "Number of Nodes: 200 (Model-6)","Number of Nodes: 50 (Main Model)"])
                    
# axarr1[0, 1].plot(x_axis1, LSTM_5[:10])
# axarr1[0, 1].plot(x_axis1, LSTM_6[:10])
# axarr1[0, 1].plot(x_axis1, LSTM_main[:10])
# axarr1[0, 1].set_xticks(x_axis1)
# axarr1[0, 1].set_title("LSTM Network",fontweight ="bold", fontsize=15)
# axarr1[0, 1].set(xlabel = "epochs", ylabel = "loss")
# axarr1[0, 1].legend(["Number of Nodes: 128 (Model-5)", "Number of Nodes: 256 (Model-6)","Number of Nodes: 512 (Main Model)"])

# axarr1[1, 0].plot(x_axis1, transformer_5[:10])
# axarr1[1, 0].plot(x_axis1, transformer_6[:10])
# axarr1[1, 0].plot(x_axis1, transformer_main[:10])
# axarr1[1, 0].set_xticks(x_axis1)
# axarr1[1, 0].set_title("Transformer Network",fontweight ="bold", fontsize=15)
# axarr1[1, 0].set(xlabel = "epochs", ylabel = "loss")
# axarr1[1, 0].legend(["Number of Nodes: 128 (Model-5)", "Number of Nodes: 256 (Model-6)","Number of Nodes: 512 (Main Model)"])

# columns = [{"Network Type":1, "Model":2, "Loss drop in 10 epochs (%)":3}]
# cell_values = [["RNN", "Main Model (Number of Nodes: 50)", str(RNN_loss_drop[0]) +" %"],
#                 ["RNN", "Model-5 (Number of Nodes: 100)", str(RNN_loss_drop[5]) +" %"],
#                 ["RNN", "Model-6 (Number of Nodes: 200)", str(RNN_loss_drop[6]) +" %"],

#                 ["LSTM", "Main Model (Number of Nodes: 512)", str(LSTM_loss_drop[0]) +" %"],
#                 ["LSTM", "Model-5 (Number of Nodes: 128)", str(LSTM_loss_drop[5]) +" %"],
#                 ["LSTM", "Model-6 (Number of Nodes: 256)", str(LSTM_loss_drop[6]) +" %"],

#                 ["Transformer", "Main Model (Number of Nodes: 512)", str(transformer_loss_drop[0]) +" %"],
#                 ["Transformer", "Model-5 (Number of Nodes: 128)", str(transformer_loss_drop[5]) +" %"],
#                 ["Transformer", "Model-6 (Number of Nodes: 256)", str(transformer_loss_drop[6]) +" %"]]

# dff_columns = pd.DataFrame(columns)
# dff_values = pd.DataFrame(cell_values)

   

# axarr1[1,1].set_axis_off() 
# table = axarr1[1,1].table( 
#     cellText = dff_values.values,  
#     # rowLabels = val2,  
#     colLabels = dff_columns.columns, 
#     # rowColours =["palegreen"] * 10,  
#     colColours =["yellow"] * 3, 
#     cellLoc ='center',  
#     loc ='upper left')         
   
# axarr1[1,1].set_title('Loss Drop Comparison (%)', 
#               fontweight ="bold", fontsize = 15) 
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.auto_set_column_width(col=list(range(len(dff_values.columns))))

# table[(1, 2)].set_facecolor("palegreen")
# table[(2, 2)].set_facecolor("palegreen")
# table[(3, 2)].set_facecolor("palegreen")
# table[(4, 2)].set_facecolor("palegreen")
# table[(5, 2)].set_facecolor("palegreen")
# table[(6, 2)].set_facecolor("palegreen")
# table[(7, 2)].set_facecolor("palegreen")
# table[(8, 2)].set_facecolor("palegreen")
# table[(9, 2)].set_facecolor("palegreen")

"NETWORK COMPARISON PLOT COMPARISON WITH LOSS DROP PERCENTAGE"


# f1, axarr1 = plt.subplots(2, 2)
# axarr1 = axarr1.reshape(2,2)
# f1.suptitle('Network Type Loss Comparisons',fontweight ="bold", fontsize=25)

# x_axis1 = list(range(RNN_main.size+1))
# x_axis1 = np.array(x_axis1[1:])
# x_axis2 = list(range(LSTM_main.size+1))
# x_axis2 = np.array(x_axis2[1:])
# x_axis3 = list(range(transformer_main.size+1))
# x_axis3 = np.array(x_axis3[1:])


# axarr1[0, 0].plot(x_axis1, RNN_main)
# axarr1[0, 0].set_xticks(x_axis1)
# axarr1[0, 0].set_title("RNN Network",fontweight ="bold", fontsize=15)
# axarr1[0, 0].set(xlabel = "epochs", ylabel = "loss")
# axarr1[0, 0].legend(["Main Model"])
                    

# axarr1[0, 1].plot(x_axis2, LSTM_main)
# axarr1[0, 1].set_xticks(x_axis2)
# axarr1[0, 1].set_title("LSTM Network",fontweight ="bold", fontsize=15)
# axarr1[0, 1].set(xlabel = "epochs", ylabel = "loss")
# axarr1[0, 1].legend(["Main Model"])


# axarr1[1, 0].plot(x_axis3, transformer_main)
# axarr1[1, 0].set_title("Transformer Network",fontweight ="bold", fontsize=15)
# axarr1[1, 0].set(xlabel = "epochs", ylabel = "loss")
# axarr1[1, 0].legend(["Main Model"])

# columns = [{"Network Type":1, "Model":2, "Loss drop in first 10 epochs (%)":3}]
# cell_values = [["RNN", "Main Model", str(RNN_loss_drop[0]) +" %"],

#                 ["LSTM", "Main Model", str(LSTM_loss_drop[0]) +" %"],

#                 ["Transformer", "Main Model", str(transformer_loss_drop[0]) +" %"]]

# dff_columns = pd.DataFrame(columns)
# dff_values = pd.DataFrame(cell_values)

   

# axarr1[1,1].set_axis_off() 
# table = axarr1[1,1].table( 
#     cellText = dff_values.values,  
#     # rowLabels = val2,  
#     colLabels = dff_columns.columns, 
#     # rowColours =["palegreen"] * 10,  
#     colColours =["yellow"] * 3, 
#     cellLoc ='center',  
#     loc ='upper left')         
   
# axarr1[1,1].set_title('Loss Drop Comparison (%)', 
#               fontweight ="bold", fontsize = 15) 
# table.auto_set_font_size(False)
# table.set_fontsize(12)
# table.auto_set_column_width(col=list(range(len(dff_values.columns))))

# table[(1, 2)].set_facecolor("palegreen")
# table[(2, 2)].set_facecolor("palegreen")
# table[(3, 2)].set_facecolor("palegreen")

"TABLE FOR TEXT GENERATION OF THE MODELS"

columns = ["Network Type", "Model", "Generated Text"]
columns = [{"Network Type":1, "Model":2, "Generated Text":3}]
cell_values = [["RNN", "Main Model", """rs," said Fhe forned mus, and a will whot were com.  All only.  As than for his at was before as the latch over they sezzy.
"Goodies it, an behing and quibls.  Harry with in Denow his unuture for amlot that says.  Thurre, and toom offey Magun the Ground to an 
Harry they ely stiff a larching the stangged woll the woald her's."I nevined wavell; Harry were trikt.
Whey wart to halmicoll but into I gut just on, of eally; in foursied insile I manicing"""],

                ["LSTM", "Main Model", """   Harry walked on her cars.  As George, Cedric Diggory, and Dumbledore didn't have to shatter 
                  when he curiously at Cedric would bring himself on a gone from the rest of the subject."You've given him my deaths, 
                  and Bagman will pay from nine no loss."The servant. . .""Seats her.  I know you find me the third task."
Hagrid began to talk to was worried.By instant, to revive him an absence.The Triwizard Cup faces in bratch, black dreams at Dumbledore's cave.
"""],

                ["Transformer", "Main Model", """all those peoples deaths were concealed had a bit out for my father kept his eye upon the color, saying me!
                  said in the dark mark? harry saw an idiot she ate grapefruit at the thing for the door opened, 
                  not look about eighty times like crouch is murmur of course, you, and they swiveled onto his firebolt, 
                  said cedric and his blind panic that sirius would be wondering whether or all in the daily prophet, 
                  the pale drowsy around its smell, you going to leave with proper name came to"""]]

dff_columns = pd.DataFrame(columns)
dff_values = pd.DataFrame(cell_values)
   
fig, ax = plt.subplots() 
ax.set_axis_off() 
table = ax.table( 
    cellText = dff_values.values,  
    # rowLabels = val2,  
    colLabels = dff_columns.columns, 
    # rowColours =["palegreen"] * 10,  
    colColours =["yellow"] * 5, 
    cellLoc ='center',  
    loc ='upper left')         
   
# ax.set_title('Text Generation Comparison', 
#               fontweight ="bold", fontsize = 25) 
table.auto_set_font_size(False)
table.set_fontsize(10)
# table.auto_set_column_width(col=list(range(len(dff_values.columns))))

table[(0, 0)].set_height(0.03)
table[(0, 0)].set_width(0.07)
table[(0, 1)].set_height(0.03)
table[(0, 1)].set_width(0.06)
table[(0, 2)].set_height(0.03)
table[(0, 2)].set_width(0.7)

table[(1, 0)].set_height(0.10)
table[(1, 0)].set_width(0.07)
table[(1, 1)].set_height(0.10)
table[(1, 1)].set_width(0.06)
table[(1, 2)].set_height(0.10)
table[(1, 2)].set_width(0.7)

table[(2, 0)].set_height(0.11)
table[(2, 0)].set_width(0.07)
table[(2, 1)].set_height(0.11)
table[(2, 1)].set_width(0.06)
table[(2, 2)].set_height(0.11)
table[(2, 2)].set_width(0.7)

table[(3, 0)].set_height(0.13)
table[(3, 0)].set_width(0.07)
table[(3, 1)].set_height(0.13)
table[(3, 1)].set_width(0.06)
table[(3, 2)].set_height(0.13)
table[(3, 2)].set_width(0.7)

# table[(2, 0)].set_facecolor("lightcoral")
# table[(3, 0)].set_facecolor("lightcoral")


   
plt.show() 