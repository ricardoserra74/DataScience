# Creamos un dataset dummy, donde hayamos transformado las variables categóricas, para poder trabajar con él cuando vayamos a tunear hiperparámetros desde fuera de Pycaret (Pycaret hace transformación de variables)
dataset_dum = dataset.drop(['CustomerID'], axis=1)

dataset_dum.Gender.replace(('Male','Female'),
                      (0,1),inplace=True)

dataset_dum.CampaignType.replace(('Conversion','Awareness', 'Consideration', 'Retention'),
                      (1,2,3,4),inplace=True)

dataset_dum.CampaignChannel.replace(('Referral','PPC', 'Email', 'SEO', 'Social Media'),
                      (1,2,3,4,5),inplace=True)



Gender_df = pd.get_dummies(dataset_dum.Gender, prefix='Gender')
Gender_df['Gender_1'] = Gender_df['Gender_1'].astype('int')
Gender_df['Gender_0'] = Gender_df['Gender_0'].astype('int')


CampaignChannel_df = pd.get_dummies(dataset_dum.CampaignChannel, prefix='Channels')
CampaignChannel_df['Channels_3'] = CampaignChannel_df['Channels_3'].astype('int')
CampaignChannel_df['Channels_2'] = CampaignChannel_df['Channels_2'].astype('int')
CampaignChannel_df['Channels_1'] = CampaignChannel_df['Channels_1'].astype('int')
CampaignChannel_df['Channels_4'] = CampaignChannel_df['Channels_4'].astype('int')
CampaignChannel_df['Channels_5'] = CampaignChannel_df['Channels_5'].astype('int')

CampaignType_df = pd.get_dummies(dataset_dum.CampaignType, prefix='Objetivo')
CampaignType_df['Objetivo_1'] = CampaignType_df['Objetivo_1'].astype('int')
CampaignType_df['Objetivo_2'] = CampaignType_df['Objetivo_2'].astype('int')
CampaignType_df['Objetivo_3'] = CampaignType_df['Objetivo_3'].astype('int')
CampaignType_df['Objetivo_4'] = CampaignType_df['Objetivo_4'].astype('int')


dataset_dum = pd.concat([Gender_df, CampaignChannel_df, CampaignType_df, dataset_dum], axis=1)

pd.options.display.max_columns = None
dataset_dum.head(5)

# Borramos variables para evitar colinealidad
dataset_dum = dataset_dum.drop(['Gender', 'Gender_1', 'Channels_1', 'Objetivo_4', 'CampaignChannel', 'CampaignType'], axis=1)

# Dividimos el dataset, que luego también usaremos
varObj_dum = dataset_dum.y 
imput_dum = dataset_dum.drop(['y'],axis=1)

# Creamos training y test para usar con Pycaret (en principio no van a ahcer falta)
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(imput, varObj, test_size=0.2, random_state=1234)

# Creamos training y test para usar sin Pycaret, para todos los modelos que generemos de aquí en adelante:
X_train_dum, X_test_dum, y_train_dum, y_test_dum = train_test_split(imput_dum, varObj_dum, test_size=0.2, random_state=1234)

# Con la librería Pycaret obtenemos los modelos: en el setup se hacen automáticamente todas las transformaciones de las variables, el split en train y test, le meto balanceo
# Por defecto: el train size es 70%, aplica data_split_stratify
model_setup = setup(data=dataset_dum, target='y', session_id=1234, fix_imbalance = True)
models()

## los primeros modelos son los ensemble: .
best_models = compare_models()

# Devuelve el mejor modelo
model_setup.automl()


