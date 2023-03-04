# Random Processes Computer Proj 2
# submitted by: Nitzan Elimelech & Gal Sarid
#
# region imports
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.ticker import MaxNLocator
from sklearn.datasets import fetch_openml
from skimage.feature import hog
from sklearn import svm
from time import time

import matplotlib.pyplot as plt
import numpy.random as rnd
import numpy as np
# endregion

# region Project Classes
class DataUtils():

  def resize_data_set(input_x, input_y, digit_images_num=1000, image_size=784):
    x = []
    y = []
    data_target_tuple = list(zip(input_x, input_y))

    for i in range(0,10):
      cnt = 0
      for dt in data_target_tuple:
        if int(dt[-1]) == i:
          x.append(dt[0])
          y.append(dt[-1])
          cnt += 1
          if cnt >= digit_images_num:
            break

    return np.array(x), np.array(y)

  def count_data_groups(x):
    tmp = np.zeros(10)
    counter = []

    for i in range(0,10):
      for xi in x:
        if int(xi) == i:
          tmp[i] += 1
      counter.append((str(i), int(tmp[i])))
    return counter

  def plot_data_distribution(data_count):
    fig, ax = plt.subplots(figsize=(13.33,7.5), dpi = 96)

    # Plot bars
    labels = list(map(np.array, zip(*data_count)))
    bar1 = ax.bar(labels[0], labels[-1], label="digits distribution", width=0.5)

    # Create the grid 
    ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
    ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)

    # Reformat x-axis label and tick labels
    ax.set_xlabel('', fontsize=12, labelpad=10)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_major_formatter(lambda s, i : str(s))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=12, labelrotation=0)
    ax.set_xticks(labels[0])

    # Reformat y-axis
    ax.set_title('MNIST data-set labels distribution', fontsize=18, weight='bold')
    ax.set_ylabel('Number of images', fontsize=12, labelpad=10)
    ax.set_xlabel('Label', fontsize=12, labelpad=10)
    ax.yaxis.set_label_position("left")
    ax.xaxis.set_major_formatter(lambda s, i : str(s))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_tick_params(pad=2, labeltop=False, labelbottom=True, bottom=False, labelsize=12)

    # Add label on top of each bar
    ax.bar_label(bar1, labels=[str(int(e)) for e in labels[-1]], padding=6, color='black', fontsize=9)

    # Find the average data points
    average = labels[-1].mean()

    # Plot average (horizontal) line
    plt.axhline(y=average, color = 'red', linewidth=1.5)

    # Determine the y-limits of the plot
    ymin, ymax = ax.get_ylim()
    # Calculate a suitable y position for the text label
    y_pos = average/ymax
    y_pos = y_pos + 0.02 if average > 6000 else y_pos - 0.02
    # Annotate the average line
    ax.text(0.615, y_pos, f'Average = {int(average)}', ha='right', va='center', transform=ax.transAxes, size=8, zorder=3, weight='bold')

    # Add legend
    #ax.legend(loc="best", ncol=2, bbox_to_anchor=[1, 1.07], borderaxespad=0, frameon=False, fontsize=8)

  def plot_trained_model_total_stats(
        trained_models, 
        gs_params, 
        is_hog, 
        graph_name=None,
        colors=['blue', 'red', 'green', 'orange']):
     fig, ax = plt.subplots()
     for tm, gs, c in zip(trained_models, gs_params, colors):
       ax.scatter(
          x=tm.train_time + gs.grid_search_time,
          y=round(tm.test_accuracy * 100, 3),
          c=c,
          label=tm.model_name,
          marker='o' if 'svm' in tm.model_name else '^'
        )
       if graph_name is None:
            graph_name = 'SVM-HOG' if is_hog else 'SVM'
            graph_name += ' trained models accuracy over total train time' 
     ax.grid()
     ax.legend()
     ax.set_title(graph_name, fontsize=18, weight='bold')
     ax.set_ylabel('Accuracy [pct]', fontsize=12, labelpad=10)
     ax.set_xlabel('Total Train Time [sec]', fontsize=12, labelpad=10)

class ImageUtils():
  
  def calc_hog_features(x, image_shape=(28, 28), pixels_per_cell=(8, 8)):
      fd_list = []
      for row in x:
          img = row.reshape(image_shape)
          fd = hog(img, orientations=8, pixels_per_cell=pixels_per_cell, cells_per_block=(1, 1))
          fd_list.append(fd)
      
      return np.array(fd_list)

  def show_radom_image(x, show_hog=True):
    idx = rnd.randint(0, len(x))
    img = x[idx].reshape((28, 28))
    plt.figure()
    plt.imshow(img, cmap='gray')

    if show_hog:
      plt.figure()
      _,hog_image = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
      plt.title(f'HOG of image number {idx+1}')
      plt.imshow(hog_image, cmap='gray')

class BestModelParams():

  def __init__(self, model_name, estimator, param_grid, x_gs, y_gs, load_known_params, is_hog):
    self.model_name = model_name
    if load_known_params:
        self.best_params = Consts.get_params_by_model_name(model_name, is_hog)
        self.best_score = Consts.get_best_score_by_model_name(model_name, is_hog)
        self.grid_search_time = Consts.get_grid_search_time_by_model_name(model_name, is_hog)
    else:
        grid = GridSearchCV(
            estimator=estimator,
            return_train_score=False,
            param_grid=param_grid, 
            scoring='accuracy', 
            cv=10, 
        )
        start = time()
        grid.fit(x_gs, y_gs)
        end = time()
        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        print('best params:')
        print(grid.best_params_)
        print(f'score: {grid.best_score_}')
        self.grid_search_time = end - start
        print(f'grid_search_time: {self.grid_search_time}')

class TrainedModel():
   
   def __init__(self, model_name, model, train_time, test_accuracy, cm):
      self.model_name = model_name
      self.model = model
      self.train_time = train_time
      self.test_accuracy = test_accuracy
      self.cm = cm

class BaseHandler():
  
  def find_best_model_params(model_names, estimator, param_grids, x, y, load_known_params, is_hog):
   x,y = DataUtils.resize_data_set(x,y, Consts.GS_SIZE)
   best_params = []
   for mn, pg in zip(model_names, param_grids):
        print(f'start process model: {mn}')
        best_params.append(BestModelParams(
                model_name=mn if not is_hog else f'{mn}-hog',
                estimator=estimator,
                param_grid=pg,
                x_gs=x,
                y_gs=y,
                load_known_params=load_known_params,
                is_hog=is_hog
            )
        )
   return best_params

class SvmHandler(BaseHandler):
   
  def handle_svm(x, y):    
    # find svm best params
    svm_best_params = SvmHandler.find_best_model_params(
            model_names=Consts.SVM_MODEL_NAMES,
            estimator=svm.SVC(),
            param_grids=Consts.PARAM_GRID_SVM,
            x=x,
            y=y,
            load_known_params=True,
            is_hog=False
        )
    
    # train svm models [resize because of long train time for full size]
    x_resized, y_resized = DataUtils.resize_data_set(x, y, Consts.SVM_SIZE)
    counter = DataUtils.count_data_groups(y_resized)
    msg = f"MNIST resized labels distribution:\n {counter}"
    print(msg)
    DataUtils.plot_data_distribution(counter)
    svm_trained_models = SvmHandler.train_svm_models(svm_best_params, x_resized, y_resized)

    # plot all data
    DataUtils.plot_trained_model_total_stats(svm_trained_models, svm_best_params, False)
    return svm_trained_models, svm_best_params

  def handle_svm_hog(x, y):
    # find svm best params for hog
    x_hog = ImageUtils.calc_hog_features(x)
    svm_hog_best_params = SvmHandler.find_best_model_params(
            model_names=Consts.SVM_MODEL_NAMES,
            estimator=svm.SVC(),
            param_grids=Consts.PARAM_GRID_SVM,
            x=x_hog,
            y=y,
            load_known_params=True,
            is_hog=True
        )

    # train svm models with hog
    hog_trained_models = SvmHandler.train_svm_models(svm_hog_best_params, x_hog, y)

    # plot all data
    DataUtils.plot_trained_model_total_stats(hog_trained_models, svm_hog_best_params, True)
    return hog_trained_models, svm_hog_best_params 

  def train_svm_models(params, x, y):
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=Consts.TEST_SIZE, shuffle=True) 
   trained_models = []
   print('start train svm models\n')
   for param in params:
        print(f'model: {param.model_name}')
        svc = Consts.get_svc_by_model_name(param.model_name, param.best_params)
        start = time()
        svc.fit(x_train, y_train)
        end = time()
        elapsed = end - start
        print(f'train time: {elapsed}')
        print(f'support vectors shape: {svc.support_vectors_.shape}')
        y_pred = svc.predict(x_test)
        score = svc.score(x_test, y_test)
        print(f'score: {score}')
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{param.model_name} confusion matrix", fontsize=16, weight='bold')
        print(f'finish process {param.model_name}\n')
        trained_models.append(TrainedModel(
                model_name=param.model_name,
                model=svc,
                train_time=elapsed,
                test_accuracy=score,
                cm=cm
           )
        )
   return trained_models

class KnnHandler(BaseHandler):
   
     def handle_knn(x, y, is_hog=False):    
        # find knn best params
        knn_best_params = KnnHandler.find_best_model_params(
                model_names=Consts.KNN_MODEL_NAMES,
                estimator=KNeighborsClassifier(),
                param_grids=Consts.PARAM_GRID_KNN,
                x=x,
                y=y,
                load_known_params=True,
                is_hog=is_hog
            )
        
        # train knn model
        knn_trained_model = KnnHandler.train_knn_model(knn_best_params, x, y)
        return [knn_trained_model], knn_best_params

     def handle_knn_hog(x, y):    
        x_hog = ImageUtils.calc_hog_features(x)
        return KnnHandler.handle_knn(x_hog, y, is_hog=True)

     def train_knn_model(params, x, y):
        param = params[0]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=Consts.TEST_SIZE, shuffle=True) 
        print('start train knn models\n')
        print(f'model: {param.model_name}')
        knn = KNeighborsClassifier(
            n_neighbors=param.best_params["n_neighbors"],
            algorithm=param.best_params["algorithm"],
            weights=param.best_params["weights"]
        )
        start = time()
        knn.fit(x_train, y_train)
        end = time()
        elapsed = end - start
        print(f'train time: {elapsed}')
        y_pred = knn.predict(x_test)
        score = knn.score(x_test, y_test)
        print(f'score: {score}')
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f"{param.model_name} confusion matrix", fontsize=16, weight='bold')
        print(f'finish process {param.model_name}\n')
        return TrainedModel(
           model_name=param.model_name,
           model=knn,
           train_time=elapsed,
           test_accuracy=score,
           cm=cm
        )

class Consts():
    SVM_MODEL_NAMES = ['svm-sigmoid', 'svm-poly', 'svm-rbf', 'svm-linear']
    KNN_MODEL_NAMES = ['knn']
    
    # used to find best params with GridSearch
    PARAM_GRID_SVM = [
        {"kernel": ["sigmoid"], "gamma": [1e-1, 1e-2, 1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["poly"], "degree": [2, 3, 4, 5, 6, 7, 8], "C": [1, 10, 100, 1000]},
        {"kernel": ["rbf"], "gamma": [1e-1, 1e-2, 1e-3, 1e-4], "C": [1, 10, 100, 1000]},
        {"kernel": ["linear"], "C": [1, 10, 100, 1000]}
    ]

    # used for runs to come after best params was found
    PARAM_SVM_BEST = [
        {"kernel": "sigmoid", "gamma": 1e-1, "C": 1},
        {"kernel": "poly", "degree": 2, "C": 100},
        {"kernel": "rbf", "gamma": 1e-3, "C": 1},
        {"kernel": "linear", "C": 1}
    ]

    SVM_LINEAR_BEST_SCORE = 0.9099
    SVM_POLY_BEST_SCORE = 0.9621
    SVM_SIGMOID_BEST_SCORE = 0.0891
    SVM_RBF_BEST_SCORE = 0.0891

    SVM_LINEAR_GS_TIME = 214.1319
    SVM_POLY_GS_TIME = 2713.8610
    SVM_SIGMOID_GS_TIME = 8010.5545
    SVM_RBF_GS_TIME = 9480.4730

    # used for runs to come after best params was found
    PARAM_SVM_HOG_BEST = [
        {"kernel": "sigmoid", "gamma": 1e-3, "C": 1000},
        {"kernel": "poly", "degree": 6, "C": 1},
        {"kernel": "rbf", "gamma": 1e-1, "C": 10},
        {"kernel": "linear", "C": 1}
    ]
    
    SVM_LINEAR_HOG_BEST_SCORE = 0.9109
    SVM_POLY_HOG_BEST_SCORE = 0.9324
    SVM_SIGMOID_HOG_BEST_SCORE = 0.9109
    SVM_RBF_HOG_BEST_SCORE = 0.9295

    SVM_LINEAR_HOG_GS_TIME = 584.1781
    SVM_POLY_HOG_GS_TIME = 411.6837
    SVM_SIGMOID_HOG_GS_TIME = 486.2844
    SVM_RBF_HOG_GS_TIME = 522.9939

    # used to find best params with GridSearch
    PARAM_GRID_KNN = [
        {
            "n_neighbors": [1, 5, 10, 15, 20, 25, 30],
            "algorithm": ["ball_tree", "kd_tree", "brute"],
            "weights": ["uniform", "distance"]
        }
    ]

    # used for runs to come after best params was found
    PARAM_KNN_BEST = {
        "n_neighbors": 5,
        "algorithm": "ball_tree",
        "weights": "distance"
    }

    KNN_BEST_SCORE = 0.9438
    KNN_GS_TIME = 3011.8065

    # used for runs to come after best params was found
    PARAM_KNN_HOG_BEST = {
        "n_neighbors": 10,
        "algorithm": "ball_tree",
        "weights": "uniform"
    }

    KNN_HOG_BEST_SCORE = 0.8716
    KNN_HOG_GS_TIME = 370.1276
    
    GS_SIZE = 1000
    SVM_SIZE = 3000
    TEST_SIZE = 0.1
    __MNIST = None

    def get_mnist_data():
       if Consts.__MNIST is None:
          Consts.__MNIST = fetch_openml('mnist_784')
       return np.array(Consts.__MNIST.data), np.array(Consts.__MNIST.target)

    def get_params_by_model_name(model_name, is_hog):
       if 'knn' in model_name:
          return Consts.PARAM_KNN_HOG_BEST if is_hog else Consts.PARAM_KNN_BEST
       params = []
       kernel = ''
       best_params = []
       # resolve kernel
       if 'svm-linear' in model_name:
          kernel = 'linear'
       elif 'svm-poly' in model_name:
          kernel = 'poly'
       elif 'svm-sigmoid' in model_name:
          kernel = 'sigmoid'
       elif 'svm-rbf' in model_name:
          kernel = 'rbf'
       # resolve params set
       params = Consts.PARAM_SVM_HOG_BEST if 'hog' in model_name else Consts.PARAM_SVM_BEST
       best_params = [p for p in params if p['kernel'] == kernel]
       return best_params[0]

    def get_best_score_by_model_name(model_name, is_hog):
       if 'svm-linear' in model_name:
            return Consts.SVM_LINEAR_HOG_BEST_SCORE if is_hog else Consts.SVM_LINEAR_BEST_SCORE
       elif 'svm-poly' in model_name:
            return Consts.SVM_POLY_HOG_BEST_SCORE if is_hog else Consts.SVM_POLY_BEST_SCORE
       elif 'svm-sigmoid' in model_name:
            return Consts.SVM_SIGMOID_HOG_BEST_SCORE if is_hog else Consts.SVM_SIGMOID_BEST_SCORE
       elif 'svm-rbf' in model_name:
            return Consts.SVM_RBF_HOG_BEST_SCORE if is_hog else Consts.SVM_RBF_BEST_SCORE
       elif 'knn' in model_name:
          return Consts.KNN_HOG_BEST_SCORE if is_hog else Consts.KNN_BEST_SCORE
       
    def get_grid_search_time_by_model_name(model_name, is_hog):
       if 'svm-linear' in model_name:
            return Consts.SVM_LINEAR_HOG_GS_TIME if is_hog else Consts.SVM_LINEAR_GS_TIME
       elif 'svm-poly' in model_name:
            return Consts.SVM_POLY_HOG_GS_TIME if is_hog else Consts.SVM_POLY_GS_TIME
       elif 'svm-sigmoid' in model_name:
            return Consts.SVM_SIGMOID_HOG_GS_TIME if is_hog else Consts.SVM_SIGMOID_GS_TIME
       elif 'svm-rbf' in model_name:
            return Consts.SVM_RBF_HOG_GS_TIME if is_hog else Consts.SVM_RBF_GS_TIME
       elif 'knn' in model_name:
          return Consts.KNN_HOG_GS_TIME if is_hog else Consts.KNN_GS_TIME
       
    def get_svc_by_model_name(model_name, params):
       if 'svm-linear' in model_name:
            return svm.SVC(
               kernel='linear',
               C=params['C']
            )
       elif 'svm-poly' in model_name:
            return svm.SVC(
               kernel='poly',
               degree=params['degree'],
               C=params['C']
            )
       elif 'svm-sigmoid' in model_name:
            return svm.SVC(
               kernel='sigmoid',
               gamma=params['gamma'],
               C=params['C']
            )
       elif 'svm-rbf' in model_name:
            return svm.SVC(
               kernel='rbf',
               gamma=params['gamma'],
               C=params['C']
            )
# endregion

# main
def main():
    # get data set and present distribution by labels
    x,y = Consts.get_mnist_data()
    counter = DataUtils.count_data_groups(y)
    msg = f"MNIST labels distribution:\n {counter}"
    print(msg)
    DataUtils.plot_data_distribution(counter)
    
    all_trained_models = []
    temp_trained_models = []
    all_gs_params = []
    temp_gs_params = []

    # run svm section
    temp_trained_models, temp_gs_params = SvmHandler.handle_svm(x, y)
    all_trained_models.extend(temp_trained_models)
    all_gs_params.extend(temp_gs_params)
    
   #  # run svm hog section
    temp_trained_models, temp_gs_params = SvmHandler.handle_svm_hog(x, y)
    all_trained_models.extend(temp_trained_models)
    all_gs_params.extend(temp_gs_params)

   #  # run knn section
    temp_trained_models, temp_gs_params = KnnHandler.handle_knn(x, y)
    all_trained_models.extend(temp_trained_models)
    all_gs_params.extend(temp_gs_params)

    # run svm hog section
    temp_trained_models, temp_gs_params = KnnHandler.handle_knn_hog(x, y)
    all_trained_models.extend(temp_trained_models)
    all_gs_params.extend(temp_gs_params)

    # plot all models score over total train time
    DataUtils.plot_trained_model_total_stats(
       trained_models=all_trained_models,
       gs_params=all_gs_params,
       is_hog=False,
       graph_name="All Trained Models Accuracy Over Total Train Time",
       colors=["red","green","blue","yellow","pink","black","orange","purple","cyan","magenta"]
    )

    plt.show()
    msg = input("press any key to exit...")

# run main
if __name__ == "__main__":
    main()