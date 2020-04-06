import os
import datetime
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import xml.etree.ElementTree as et
import re

from time import time
from keras.preprocessing.image import ImageDataGenerator, Iterator, array_to_img, img_to_array, load_img
from keras.models import load_model
from darkflow.net.build import TFNet
from matplotlib.patches import Rectangle

class AnnotationParser():
    """
    The desired labels are: (l_toe, l_ankle, l_knee, l_hip, r_toe, r_ankle, r_knee, r_hip), and activity 
    with attribute activity type radio button with values: walking, standing, running, none.
    
    Example:
    --------
    >>> annotation_file_name = "../data/lore/lore.xml"
    >>> annotation_parser = AnnotationParser()
    >>> annotation_df = annotation_parser.parse(annotation_file_name=annotation_file_name)
    >>> # Showing Markers on image
    >>> # Get general video info
    >>> video_file_name = "../data/lore/lore.mp4"
    >>> cap = cv2.VideoCapture(video_file_name)
    >>> fps = cap.get(cv2.CAP_PROP_FPS)
    >>> frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    >>> cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
    >>> video_duration = cap.get(cv2.CAP_PROP_POS_MSEC)
    >>> # Extract specific frame
    >>> frame_counter = 991
    >>> annotation_subset = annotation_df.loc[annotation_df['frame']==frame_counter]
    >>> cap = cv2.VideoCapture(video_file_name)
    >>> cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter-1)
    >>> ret, frame = cap.read()
    >>> if ret == True:
    >>>     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    >>> # Showing Image
    >>> plt.figure()
    >>> plt.imshow(frame)
    >>> plt.plot(annotation_subset['x_coord'],annotation_subset['y_coord'],'*')
    >>> plt.title(annotation_subset.iloc[0]['activity'])
    """
    def __init__(self):
        pass
        
    def parse(self, annotation_file_name):
        xtree = et.parse(annotation_file_name)
        self.xroot = xtree.getroot()
        annotations = self.parse_joints()
        self.activity_markers = self.parse_activities()
        coords = annotations.apply(lambda x: self.compute_coords(x),axis=1)
        annotations = pd.concat([annotations,coords], axis=1)
        annotations['activity'] = annotations['frame'].apply(self.return_activity)
        return annotations
    
    def parse_joints(self):
        annotations = []
        for node in self.xroot.findall("track"):
            annotation_id = node.attrib.get("id")
            annotation_label = node.attrib.get("label")
            boxes = node.findall("box")
            if node.attrib['label']!='activity':
                for box in boxes:
                    annotations.append({**box.attrib,"category_id":annotation_id,"label":annotation_label})
        annotations = pd.DataFrame(annotations)
        annotations[['frame']] = annotations[['frame']].astype(int)
        annotations[['xbr','xtl','ybr','ytl']] = annotations[['xbr','xtl','ybr','ytl']].astype(float)
        return annotations
   
    def parse_activities(self):
        activity_markers = []
        for node in self.xroot.findall("track"):
            annotation_id = node.attrib.get("id")
            annotation_label = node.attrib.get("label")
            boxes = node.findall("box")
            if node.attrib['label']=='activity':
                for box in boxes:
                    box.findall('attribute')
                    activity_markers.append({'frame':box.attrib['frame'],'activity':box.findall('attribute')[0].text})        
        activity_markers = pd.DataFrame(activity_markers)
        activity_markers[['frame']] = activity_markers[['frame']].astype(int)
        activity_markers.sort_values('frame',inplace=True)
        return activity_markers
    
    def return_activity(self, frame):
        activity_index =  (self.activity_markers['frame'].values <= frame).sum() - 1
        activity = self.activity_markers.iloc[activity_index]['activity']
        activity = pd.Series(activity)
        return activity
    
    def compute_coords(self, x):
        result = pd.Series({'x_coord':(0.5*x['xbr']+0.5*x['xtl']),'y_coord':(0.5*x['ybr']+0.5*x['ytl'])})
        return result
    
    def parse_activities_only(self, annotation_file_name):
        xtree = et.parse(annotation_file_name)
        self.xroot = xtree.getroot()
        activity_markers = self.parse_activities()
        return  activity_markers

class Yolo:
    """
    Example:
    --------
    base_path = "../data/lore"
    filename = "lore_109.jpg"
    yolo_detector = Yolo()
    image_path = "{}/{}".format(base_path, filename)
    imgcv = cv2.imread(image_path)
    
    # draw boxes
    yolo_detector.draw_boxes(imgcv)
    # extract input X
    X, width_scale, height_scale, cutting_rect = yolo_detector.extract_person(imgcv)
    plt.figure()
    plt.imshow(cv2.cvtColor(img_to_array(X)/255, cv2.COLOR_BGR2RGB))
    """
    def __init__(self):
        options = {"model": "../darkflow/cfg/yolov2-tiny.cfg", 
           "load": "../darkflow/bin/yolov2-tiny.weights", 
           "threshold": 0.001,
           "config":"../darkflow/cfg/"} 
        self.tfnet = TFNet(options)
    
    def filter_people(self, imgcv):
        result = self.detect(imgcv)
        result_df=pd.DataFrame(result)
        people_mask=result_df['label']=='person'
        people_df = result_df.loc[people_mask]
        return people_df
        
    def detect(self, imgcv):
        result = self.tfnet.return_predict(imgcv)
        return result
    
    def draw_boxes(self, imgcv):
        people_df = self.filter_people(imgcv)
        im_rgb = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
        fig = plt.figure()
        plt.imshow(im_rgb)
        ax = plt.gca()
        # draw boxes
        for i in range(people_df.shape[0]):
            # draw one box
            person_result = people_df.iloc[i]
            width = person_result["bottomright"]['x'] - person_result["topleft"]['x']
            height = person_result["bottomright"]['y'] - person_result["topleft"]['y']
            rect = Rectangle((person_result["topleft"]['x'],person_result["topleft"]['y']), width, height, fill=False, color='green')
            ax.add_patch(rect)
            label = "%s (%.3f)" % (person_result['label'], person_result['confidence'])
            plt.text(person_result["topleft"]['x'], person_result["topleft"]['y'], label, color='green')
        # Plotting the maximum rect in blue
        person_result = people_df.loc[people_df['confidence'] == people_df['confidence'].max()].iloc[0]
        width = person_result["bottomright"]['x'] - person_result["topleft"]['x']
        height = person_result["bottomright"]['y'] - person_result["topleft"]['y']
        rect = Rectangle((person_result["topleft"]['x'],person_result["topleft"]['y']), width, height, fill=False, color='blue')
        ax.add_patch(rect)
        label = "%s (%.3f)" % (person_result['label'], person_result['confidence'])
        plt.text(person_result["topleft"]['x'], person_result["topleft"]['y'], label, color='blue')
    
    def extract_person(self, imgcv, target_width=192, target_height=256):
        img = array_to_img(imgcv)
        people_df = self.filter_people(imgcv)
        people_series = people_df.loc[people_df['confidence'] == people_df['confidence'].max()].iloc[0]
        x = np.array([people_series['topleft']['x'],people_series['bottomright']['x']])
        y = np.array([people_series['topleft']['y'],people_series['bottomright']['y']])
        x = np.minimum(np.maximum((2*(x - np.mean(x)) + np.mean(x)).astype('int'),0),img.size[0])
        y = np.minimum(np.maximum((1.5*(y - np.mean(y)) + np.mean(y)).astype('int'),0),img.size[1])
        cutting_rect = [x[0],y[0],x[1],y[1]]
        X = img.crop(cutting_rect)
        width_scale, height_scale = target_width/X.size[0], target_height/X.size[1]
        X = X.resize((target_width,target_height))
        return X, width_scale, height_scale, cutting_rect

class FrameExtractor():
    '''
    Class used for extracting frames from a video file.
    '''
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
        self.yolo_detector = None
        
    def get_video_duration(self):
        duration = self.n_frames/self.fps
        print('Duration: {}'.format(datetime.timedelta(seconds=duration)))
        
    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print('Extracting every {} (nd/rd/th) frame would result in {} images.'.format(every_x_frame,n_images))
        
    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext = '.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print('Created the following directory: {}'.format(dest_path))
        
        frame_cnt = 0
        img_cnt = 0

        while self.vid_cap.isOpened():
            
            success, image = self.vid_cap.read() 
            
            if not success:
                break
            
            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ''.join([img_name, '_', str(img_cnt), img_ext]))
                cv2.imwrite(img_path, image)  
                img_cnt += 1
                
            frame_cnt += 1
        
        self.vid_cap.release()
        
    def extract_frame_list(self, frame_counters, dest_path='.', img_name=None, img_ext='.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print('Created the following directory: {}'.format(dest_path))
        
        for frame_counter in frame_counters:
            self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
            ret, image = self.vid_cap.read()
            img_path = os.path.join(dest_path, ''.join([img_name, '_', str(frame_counter), img_ext]))
            cv2.imwrite(img_path, image)
            
    def extract_person_from_frame_list(self, frame_counters, dest_path='.', img_name=None, img_ext='.jpg'):
        if not self.yolo_detector:
            self.yolo_detector = Yolo()
        
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print('Created the following directory: {}'.format(dest_path))
        
        cutting_details = []
        for frame_counter in frame_counters:
            print('Extracting frame {} on {}'.format(frame_counter, self.video_path))
            self.vid_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)
            ret, image = self.vid_cap.read()
            X, width_scale, height_scale, cutting_rect = self.yolo_detector.extract_person(imgcv=image)
            img_path = os.path.join(dest_path, ''.join([img_name, '_', str(frame_counter), img_ext]))
            cv2.imwrite(img_path, img_to_array(X))
            video_name = os.path.basename(self.video_path)
            cutting_details.append({"video_name":video_name,
                                    "frame":frame_counter,
                                    "width_scale": width_scale, 
                                    "height_scale": height_scale, 
                                    "cutting_rect": cutting_rect})
        return cutting_details

class PoseIterator(Iterator):

    def __init__(self, list_IDs, annotations_df, cutting_details_df, path_to_images, 
                 batch_size=32,dim_input=(256, 192), dim_output=(78, 62), n_channels_input=3, 
                 n_channels_output=8, shuffle=True, seed=0, **kwargs):
        
        """
        Example
        -------
        >>> from sklearn.model_selection import train_test_split
        >>> data_names = data_names_df.apply(lambda x:"{}_{}.jpg".format(x['video_name'].split('.')[0],x['frame']),axis=1).values
        >>> list_IDs_train, list_IDs_test = train_test_split(data_names, test_size=0.15)
        >>> image_data_generator_params ={
        >>>     "featurewise_center":True,
        >>>     "featurewise_std_normalization":True,
        >>>     "rotation_range":5,
        >>>     "width_shift_range":0.1,
        >>>     "height_shift_range":0,
        >>>     "horizontal_flip":True}
        >>> iterator_params = {'dim_input': (256, 192),
        >>>                    'dim_output': (78, 62),
        >>>                    'batch_size': 32,
        >>>                    'n_channels_input': 3,
        >>>                    'n_channels_output': 8,
        >>>                    'shuffle': True}
        >>> path_to_images='../data_rf/images/'
        >>> it = PoseIterator(list_IDs_train, annotations_df, cutting_details_df, path_to_images, 
        >>>                  batch_size=32,dim_input=(256, 192), dim_output=(78, 62), n_channels_input=3, 
        >>>                  n_channels_output=8, shuffle=True, seed=0, **image_data_generator_params)
        >>> X,Y = it.next()
        >>> for sample_ind in range(32):
        >>>     f, ax = plt.subplots(2,4,figsize=(16,8))
        >>>     for joint_ind in range(8):
        >>>         XX = cv2.resize(X[sample_ind,:,:,:], dsize=(62,78), interpolation=cv2.INTER_CUBIC)
        >>>         ax[joint_ind//4][joint_ind%4].imshow(XX)
        >>>         ax[joint_ind//4][joint_ind%4].imshow(Y[sample_ind,:,:,joint_ind],alpha=0.5)
        >>>         ax[joint_ind//4][joint_ind%4].set_title('{}'.format(it.ORDERED_LABELS[joint_ind]))
        >>>     plt.show()
        """
        n = len(list_IDs)
        super().__init__(n, batch_size, shuffle, seed)
    
        self.list_IDs = list_IDs
        self.annotations_df = annotations_df
        self.cutting_details_df = cutting_details_df
        self.path_to_images = path_to_images
        self.batch_size = batch_size
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_channels_input = n_channels_input
        self.n_channels_output = n_channels_output
        self.shuffle = shuffle
        self.on_epoch_end()
        self.ORDERED_LABELS = ['r_toe','l_toe','r_ankle','l_ankle','r_knee','l_knee','r_hip','l_hip']
    
        # Here is our beloved image augmentator <3
        self.generator = ImageDataGenerator(**kwargs)
    
    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples from array of indices"""
        
        batch_x = np.empty((self.batch_size, *self.dim_input, self.n_channels_input))
        batch_y = np.empty((self.batch_size, *self.dim_output, self.n_channels_output))
    
        ## Generate data
        list_IDs_temp = self.list_IDs[index_array]
        for i, ID in enumerate(list_IDs_temp):
            # Getting X            
            line = ID.split('.')[:-1][0]
            regex_video = re.compile(r'_\d+$')
            regex_frame = re.compile(r'_')
            video_name = regex_video.split(line)[0]
            frame = regex_frame.split(line)[-1]
            image_path = "{}/{}".format(self.path_to_images, ID)
            XX = img_to_array(load_img(image_path))/255
            # Getting Heat Maps
            annotation = self.annotations_df.loc[(self.annotations_df['video_name']=='{}.mp4'.format(video_name))
                                           & (self.annotations_df['frame']==int(frame))]
            cutting_details = self.cutting_details_df.loc[(self.cutting_details_df['video_name']=='{}.mp4'.format(video_name))
                                           & (self.cutting_details_df['frame']==int(frame))]
            width_scale = cutting_details['width_scale'].iloc[0]
            height_scale = cutting_details['height_scale'].iloc[0]
            cutting_rect = cutting_details['cutting_rect'].iloc[0]
            joint_coords_df = annotation.set_index('label').loc[self.ORDERED_LABELS]
            joint_coords = joint_coords_df[['x_coord','y_coord']].values
            ## fixing the joint_coords after croping
            joint_coords = joint_coords - cutting_rect[:2]
            ## fixing the joint_coords after resizing
            joint_coords = np.dot(joint_coords, np.diag([width_scale, height_scale]))
            joint_coords = joint_coords.astype(int)
            # Store input
            batch_x[i,] = XX
            # Store output
            batch_y[i,] = self.get_heat_maps(joint_coords)
        
        # Transform the inputs and correct the outputs accordingly
        for i, (x, y) in enumerate(zip(batch_x, batch_y)):           
            transform_params = self.generator.get_random_transform(x.shape)
            batch_x[i] = self.generator.apply_transform(x, transform_params)
            y = cv2.resize(y, dsize=(self.dim_input[1],self.dim_input[0]), interpolation=cv2.INTER_CUBIC)
            y = self.generator.apply_transform(y, transform_params)
            batch_y[i] = cv2.resize(y, dsize=(self.dim_output[1],self.dim_output[0]), interpolation=cv2.INTER_CUBIC)
                
        return batch_x, batch_y
    
    def get_heat_maps(self, y, img_width=192, img_height=256, hmap_width=62, hmap_height=78):
        """
        Returns the 2D heatmaps associated to the coordinates specified in the rows of y, keeping the proportions
        from image to heatmap.

        Example:
        --------
        >>> joint_count = 2
        >>> y = np.hstack([np.random.randint(low=1,high=img_height,size=(joint_count,1)),
                   np.random.randint(low=1,high=img_width,size=(joint_count,1))])
        >>> H = get_heat_maps(y)
        >>> for i in range(y.shape[0]):
        >>>     plt.figure()
        >>>     plt.imshow(H[:,:,i])
        """
        joint_count = y.shape[0]
        width_scale = hmap_width/img_width;
        height_scale = hmap_height/img_height;
        xh, xw = np.meshgrid(range(hmap_height),range(hmap_width))
        H = np.zeros((hmap_height, hmap_width, joint_count))
        for joint_index in range(joint_count):
            std_h = 4*height_scale
            std_w = 4*width_scale
            H[:,:,joint_index] = np.exp(-((xh-height_scale*y[joint_index,1])/std_h)**2
                                        -((xw-width_scale*y[joint_index,0])/std_w)**2).T
        return H
    
class PoseEstimator:
    #old weights: checkpoints/simple_baseline_weights-55-0.00001780-0.00005117.h5
    def __init__(self, model_path="checkpoints/simple_baseline_weights-47-0.00001083-0.00003969.h5"):
        self.yolo_detector = Yolo()
        self.model = load_model(model_path)
        self.ORDERED_LABELS = ['r_toe','l_toe','r_ankle','l_ankle','r_knee','l_knee','r_hip','l_hip']
        
    def estimate_pose(self, imgcv):
        """
        Example:
        --------
        >>> pose_estimator = PoseEstimator()
        >>> joint_coords = pose_estimator.estimate_pose(imgcv)
        >>> f, ax = plt.subplots(1,1,figsize=(16/1.5,9/1.5))
        >>> image = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
        """
        X, width_scale, height_scale, cutting_rect = self.yolo_detector.extract_person(imgcv=imgcv)
        X = cv2.cvtColor(img_to_array(X)/255, cv2.COLOR_BGR2RGB)
        X = np.expand_dims(X, axis=0)
        # Running joint coords estimation
        y_pred_1 = self.model.predict(X)
        X_flip = np.flip(X,2)
        y_pred_2 = np.flip(self.model.predict(X_flip),2)
        y_pred = (y_pred_1 + y_pred_2)/2
        joint_coords = np.array([])        
        for joint_ind in range(8):
            # Respect Original Frame
            coords = np.where(y_pred==np.max(y_pred[0,:,:,joint_ind]))
            coords = [coord[0] for coord in coords][1:3]
            coords_1 = np.dot(np.diag([256/78,192/62]),coords)
            coords_2 = np.dot(np.diag([1/height_scale,1/width_scale]),coords_1).astype(int)
            coords = [cutting_rect[1]+coords_2[0],cutting_rect[0]+coords_2[1]]
            if len(joint_coords) == 0:
                joint_coords = [coords]
            else:
                joint_coords = np.vstack([joint_coords, [coords]])            
        return joint_coords
