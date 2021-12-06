# 1. Load Data..

nc_dir = "File Path Write Here.."
nc_files = glob.glob(nc_dir+"*.nc")
file = nc4.Dataset(nc_files[0], 'r', format='NETCDF4')

coordinates = ['latitude', 'longitude']
radiances = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 
             'ev_1km_emissive_29', 'ev_1km_emissive_33', 
             'ev_1km_emissive_34', 'ev_1km_emissive_35', 
             'ev_1km_emissive_36', 'ev_1km_refsb_26', 
             'ev_1km_emissive_27', 'ev_1km_emissive_20', 
             'ev_1km_emissive_21', 'ev_1km_emissive_22', 
             'ev_1km_emissive_23']

properties = ['cloud_water_path', 'cloud_optical_thickness', 
              'cloud_effective_radius', 'cloud_phase_optical_properties', 
              'cloud_top_pressure', 'cloud_top_height', 
              'cloud_top_temperature', 'cloud_emissivity', 
              'surface_temperature']

rois = 'cloud_mask'
labels = 'cloud_layer_type'


nc_dir = Path(nc_dir)

save_dir = Path("C:\\Users\\ernsb\Desktop\\AI FOR EARTH\\AI Projeler\\bulutt\\")
save_dir.mkdir(parents=True, exist_ok=True)

nc_paths = nc_dir.glob("*.nc")

for filename in tqdm(nc_paths):
    
    # load swath variables and label masks
    f_radiances, f_properties, f_cloud_mask, f_labels, *_ = read_nc(filename)
    
    # labelled pixels have at least one non-zero value over the vertical axis
    f_label_mask = np.sum(~f_labels.mask, 3) > 0
    
    # for the purposes of this tutorial, we are going to extract only labelled tiles
    try:
        labelled_tiles, labelled_positions = extract_cloudy_labelled_tiles((f_radiances, f_properties, f_cloud_mask, f_labels), f_cloud_mask[0], f_label_mask[0])
        
        name = os.path.basename(filename).replace(".nc", ".npz")

        np.savez_compressed(save_dir / name, 
                            radiances=labelled_tiles[0].data, 
                            properties=labelled_tiles[1].data, 
                            cloud_mask=labelled_tiles[2].data, 
                            labels=labelled_tiles[3].data, 
                            location=labelled_positions)
    
    except:
        pass
   
  dataset = CumuloDataset(root_dir="Give File Path", ext="npz")
  
  xs, ys, props = [], [], []

for filename, radiances, properties, cloud_mask, labels in dataset:
    xs.append(radiances) # radiances
    ys.append(labels) # labels
    
    props.append(properties) # we load also the physical properties and use them later on for physical evaluation

shape = xs[0].shape
X = np.vstack(xs).reshape(-1, shape[1] * shape[2] * shape[3]) # vectorize tiles
y = np.hstack(ys)

shape = props[0].shape
p = np.vstack(props).reshape(-1, shape[1] * shape[2] * shape[3]) # vectorize tiles

# 9. Elde Ettiğimiz Verilerin Boyutlarına Bakalim..
print("Number of Input Luminosities for the Model..:", X.shape)
print("Cloud Tags as Output for Model..:",y.shape)
print("Physical Properties Used to Evaluate the Results at the Physical Level...:",p.shape)

# we use 20% of data as test set
train_xs, test_xs, train_ys, test_ys = train_test_split(X, y, test_size=0.20, random_state=42)

# we use 10% of the remaining data for validation
train_xs, val_xs, train_ys, val_ys = train_test_split(train_xs, train_ys, test_size=0.10, random_state=42)

print(train_xs.shape, train_ys.shape, val_xs.shape, val_ys.shape, test_xs.shape, test_ys.shape)

#LightGBM Process
lgb_train = lgb.Dataset(train_xs, train_ys)
lgb_valid = lgb.Dataset(val_xs, val_ys)

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_classes': 8,
    'num_iterations': 400,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'verbose': 0,
}

gbm = lgb.train(params, lgb_train, valid_sets=[lgb_valid])

train_prob_pred = gbm.predict(train_xs, num_iteration = gbm.best_iteration)
val_prob_pred = gbm.predict(val_xs, num_iteration = gbm.best_iteration)
test_prob_pred = gbm.predict(test_xs, num_iteration = gbm.best_iteration)


print("Train_prob_pred Shape ..:",train_prob_pred.shape)
print("Val_prob_pred Shape ..:",val_prob_pred.shape)
print("Test_prob_pred Shape ..:",test_prob_pred.shape)

train_y_pred = np.argmax(train_prob_pred, 1)
val_y_pred = np.argmax(val_prob_pred, 1)
test_y_pred = np.argmax(test_prob_pred, 1)


print("Train_y_pred Shape..:",train_y_pred.shape)
print("Val_y_pred Shape..:",val_y_pred.shape)
print("Test_y_pred Shape..:",test_y_pred.shape)

# Confusion Matrix

train_cm = confusion_matrix(train_ys, train_y_pred, 
                            labels=range(8), normalize='true')
val_cm = confusion_matrix(val_ys, val_y_pred, 
                          labels=range(8), normalize='true')
test_cm = confusion_matrix(test_ys, test_y_pred, 
                           labels=range(8), normalize='true')

# HeatMap..
plt.figure(figsize = (20,5))

for i, (label, cm) in enumerate(zip(["TRAINING", "VALIDATION", "TEST"], [train_cm, val_cm, test_cm])):
    plt.subplot(131+i)
   

    df_cm = pd.DataFrame(cm, index=range(8), columns=range(8))

    plt.title(label)
    ax = sn.heatmap(df_cm, annot=True, vmin=0, vmax=1)
    ax.set(xlabel='predicted', ylabel='target')
 
# Evaluation of the Model
accuracy = accuracy_score(test_ys, test_y_pred)
f1 = f1_score(test_ys, test_y_pred, average='macro')

print("Test accuracy:", accuracy)
print("Test f1 score:", f1)
