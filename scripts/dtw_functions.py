# loading in some basic packages
import numpy as np                           # basic data operations
import pandas as pd                          # data wrangling
import os                                    # for foldering  
import itertools                             # for list combinations  
from tqdm import tqdm                        # for progress bars

# loading packages for dynamic time warpingfrom dtw import dtw
from dtw import dtw


def get_keypoints(original=True):
    # identify which column indices in a dataframe correspond to which body parts
    keypoint_thumb_left = ["X_LEFT_THUMB_TIP", "Y_LEFT_THUMB_TIP", "Z_LEFT_THUMB_TIP"]
    keypoint_index_left = ["X_LEFT_INDEX_FINGER_TIP", "Y_LEFT_INDEX_FINGER_TIP", "Z_LEFT_INDEX_FINGER_TIP"]
    keypoint_middle_left = ["X_LEFT_MIDDLE_FINGER_TIP", "Y_LEFT_MIDDLE_FINGER_TIP", "Z_LEFT_MIDDLE_FINGER_TIP"]
    keypoint_ring_left = ["X_LEFT_RING_FINGER_TIP", "Y_LEFT_RING_FINGER_TIP", "Z_LEFT_RING_FINGER_TIP"]
    keypoint_pinky_left = ["X_LEFT_PINKY_FINGER_TIP", "Y_LEFT_PINKY_FINGER_TIP", "Z_LEFT_PINKY_FINGER_TIP"]

    keypoint_thumb_right = ["X_RIGHT_THUMB_TIP", "Y_RIGHT_THUMB_TIP", "Z_RIGHT_THUMB_TIP"]
    keypoint_index_right =  ["X_RIGHT_INDEX_FINGER_TIP", "Y_RIGHT_INDEX_FINGER_TIP", "Z_RIGHT_INDEX_FINGER_TIP"]
    keypoint_middle_right = ["X_RIGHT_MIDDLE_FINGER_TIP", "Y_RIGHT_MIDDLE_FINGER_TIP", "Z_RIGHT_MIDDLE_FINGER_TIP"]
    keypoint_ring_right = ["X_RIGHT_RING_FINGER_TIP", "Y_RIGHT_RING_FINGER_TIP", "Z_RIGHT_RING_FINGER_TIP"]
    keypoint_pinky_right =  ["X_RIGHT_PINKY_FINGER_TIP", "Y_RIGHT_PINKY_FINGER_TIP", "Z_RIGHT_PINKY_FINGER_TIP"]

    if original:
        keypoint_wrist_left = ["X_LEFT_WRIST", "Y_LEFT_WRIST", "Z_LEFT_WRIST"]
        keypoint_wrist_right = ["X_RIGHT_WRIST", "Y_RIGHT_WRIST", "Z_RIGHT_WRIST"]
    else:
        keypoint_wrist_left = ["x_left_wrist", "y_left_wrist"]
        keypoint_wrist_right = ["x_right_wrist", "y_right_wrist"] 

    # make one 1 list with all keypoints
    keypoints = list(itertools.chain(keypoint_thumb_left, keypoint_index_left, keypoint_middle_left,
                                     keypoint_ring_left, keypoint_pinky_left, keypoint_wrist_left,
                                     keypoint_thumb_right, keypoint_index_right, keypoint_middle_right, 
                                     keypoint_ring_right, keypoint_pinky_right, keypoint_wrist_right))
    return keypoints


def get_relative_keypoints():
    # identify which column indices in a dataframe correspond to which body parts
    # keypoint_thumb_right = ["x_right_thumb_tip_relative", "y_right_thumb_tip_relative"]
    # keypoint_thumb_left = ["x_left_thumb_tip_relative", "y_left_thumb_tip_relative"]
    # keypoint_index_right =  ["x_right_index_finger_tip_relative", "y_right_index_finger_tip_relative"]
    # keypoint_index_left = ["x_left_index_finger_tip_relative", "y_left_index_finger_tip_relative"]
    # keypoint_middle_right = ["x_right_middle_finger_tip_relative", "y_right_middle_finger_tip_relative"]
    # keypoint_middle_left = ["x_left_middle_finger_tip_relative", "y_left_middle_finger_tip_relative"]
    # keypoint_ring_right = ["x_right_ring_finger_tip_relative", "y_right_ring_finger_tip_relative"]
    # keypoint_ring_left = ["x_left_ring_finger_tip_relative", "y_left_ring_finger_tip_relative"]
    # keypoint_pinky_right =  ["x_right_pinky_finger_tip_relative", "y_right_pinky_finger_tip_relative"]
    # keypoint_pinky_left = ["x_left_pinky_finger_tip_relative", "y_left_pinky_finger_tip_relative"]
    # keypoint_wrist_right = ["x_right_wrist", "y_right_wrist"] 
    # keypoint_wrist_left = ["x_left_wrist", "y_left_wrist"]

    keypoint_thumb_left = ["x_left_thumb_tip_relative", "y_left_thumb_tip_relative", "z_left_thumb_tip_relative"]
    keypoint_index_left = ["x_left_index_finger_tip_relative", "y_left_index_finger_tip_relative", "z_left_index_finger_tip_relative"]
    keypoint_middle_left = ["x_left_middle_finger_tip_relative", "y_left_middle_finger_tip_relative", "z_left_middle_finger_tip_relative"]
    keypoint_ring_left = ["x_left_ring_finger_tip_relative", "y_left_ring_finger_tip_relative", "z_left_ring_finger_tip_relative"]
    keypoint_pinky_left = ["x_left_pinky_finger_tip_relative", "y_left_pinky_finger_tip_relative", "z_left_pinky_finger_tip_relative"]
    keypoint_wrist_left = ["x_left_wrist", "y_left_wrist", "z_left_wrist"]

    keypoint_thumb_right = ["x_right_thumb_tip_relative", "y_right_thumb_tip_relative", "z_right_thumb_tip_relative"]
    keypoint_index_right =  ["x_right_index_finger_tip_relative", "y_right_index_finger_tip_relative", "z_right_index_finger_tip_relative"]
    keypoint_middle_right = ["x_right_middle_finger_tip_relative", "y_right_middle_finger_tip_relative", "z_right_middle_finger_tip_relative"]
    keypoint_ring_right = ["x_right_ring_finger_tip_relative", "y_right_ring_finger_tip_relative", "z_right_ring_finger_tip_relative"]
    keypoint_pinky_right =  ["x_right_pinky_finger_tip_relative", "y_right_pinky_finger_tip_relative", "z_right_pinky_finger_tip_relative"]
    keypoint_wrist_right = ["x_right_wrist", "y_right_wrist", "z_right_wrist"]

    # make one 1 list with all keypoints
    keypoints_relative = list(itertools.chain(keypoint_thumb_left, keypoint_index_left, keypoint_middle_left,
                                            keypoint_ring_left, keypoint_pinky_left, keypoint_wrist_left,
                                            keypoint_thumb_right, keypoint_index_right, keypoint_middle_right, 
                                            keypoint_ring_right, keypoint_pinky_right, keypoint_wrist_right))

    return keypoints_relative


#make a function that takes in two timeseries and produces a normalized dtw distance
def dtw_distance_normalized(ts1, ts2):
    ts1 = np.array(ts1)
    ts2 = np.array(ts2)
    
    #now calculate a multidimesnional (dependent) dtw distance using dtwParallel Multivariate = true, type dtw = dependent, distance is euclidean, compare the two timeseries only locally (sakoe_chiba)
    # res = dtw(ts1, ts2, step_pattern = "asymmetric", open_end = True, open_begin = True)
    res = dtw(ts1, ts2, step_pattern = "symmetric2")
    normalized_distance = res.normalizedDistance
    
    return normalized_distance


# make a dependent dtw such that each keypoint dtw distance is added up and divided by the number of keypoints
def dtw_distance_dependent(MT1, MT2, MT_B_flipped, reference, distance_array, hands_dtw, use_mirrored_ts, keypoints):
    ### initialize all distance variables with NaN
    dis_left_thumb, dis_left_index, dis_left_middle, dis_left_ring, dis_left_pinky, dis_left_wrist = [np.nan]*6
    dis_right_thumb, dis_right_index, dis_right_middle, dis_right_ring, dis_right_pinky, dis_right_wrist = [np.nan]*6
    dis_left_right_thumb, dis_left_right_index, dis_left_right_middle, dis_left_right_ring, dis_left_right_pinky, dis_left_right_wrist = [np.nan]*6
    dis_right_left_thumb, dis_right_left_index, dis_right_left_middle, dis_right_left_ring, dis_right_left_pinky, dis_right_left_wrist = [np.nan]*6
    distance_left, distance_right, distance_left_mirrored, distance_right_mirrored, distance_left_right, distance_right_left = [np.nan]*6

    ### keypoints
    keypoint_thumb_left = keypoints[0:3]
    keypoint_index_left = keypoints[3:6]
    keypoint_middle_left = keypoints[6:9]
    keypoint_ring_left = keypoints[9:12]
    keypoint_pinky_left = keypoints[12:15]
    keypoint_wrist_left = keypoints[15:18]
    keypoint_thumb_right = keypoints[18:21]
    keypoint_index_right = keypoints[21:24]
    keypoint_middle_right = keypoints[24:27]
    keypoint_ring_right = keypoints[27:30]
    keypoint_pinky_right = keypoints[30:33]
    keypoint_wrist_right = keypoints[33:36]

    if hands_dtw == "both" or hands_dtw == "left" in hands_dtw:
        ### xyz
        dis_left_thumb = dtw_distance_normalized(MT1[keypoint_thumb_left], MT2[keypoint_thumb_left])
        dis_left_index = dtw_distance_normalized(MT1[keypoint_index_left], MT2[keypoint_index_left])
        dis_left_middle = dtw_distance_normalized(MT1[keypoint_middle_left], MT2[keypoint_middle_left])
        dis_left_ring = dtw_distance_normalized(MT1[keypoint_ring_left], MT2[keypoint_ring_left])
        dis_left_pinky = dtw_distance_normalized(MT1[keypoint_pinky_left], MT2[keypoint_pinky_left])
        dis_left_wrist = dtw_distance_normalized(MT1[keypoint_wrist_left], MT2[keypoint_wrist_left])
        ## calculate the mean distance for the hands
        distance_left = np.mean([dis_left_thumb, dis_left_index, dis_left_middle, 
                                    dis_left_ring, dis_left_pinky, dis_left_wrist])

    if hands_dtw == "both" or hands_dtw == "right" in hands_dtw:
        ### xyz
        dis_right_thumb = dtw_distance_normalized(MT1[keypoint_thumb_right], MT2[keypoint_thumb_right])
        dis_right_index = dtw_distance_normalized(MT1[keypoint_index_right], MT2[keypoint_index_right])
        dis_right_middle = dtw_distance_normalized(MT1[keypoint_middle_right], MT2[keypoint_middle_right])
        dis_right_ring = dtw_distance_normalized(MT1[keypoint_ring_right], MT2[keypoint_ring_right])
        dis_right_pinky = dtw_distance_normalized(MT1[keypoint_pinky_right], MT2[keypoint_pinky_right])
        dis_right_wrist = dtw_distance_normalized(MT1[keypoint_wrist_right], MT2[keypoint_wrist_right])
        ## calculate the mean distance for the hands
        distance_right = np.mean([dis_right_thumb, dis_right_index, dis_right_middle, 
                                dis_right_ring, dis_right_pinky, dis_right_wrist])
    
    ### calculate the distance for opposite hands (left thumb with right thumb, left index with right index, etc.)
    if "_" in hands_dtw: # if hands_dtw is left_right or right_left
        ### xyz
        dis_left_right_thumb = dtw_distance_normalized(MT1[keypoint_thumb_left], MT2[keypoint_thumb_right])
        dis_left_right_index = dtw_distance_normalized(MT1[keypoint_index_left], MT2[keypoint_index_right])
        dis_left_right_middle = dtw_distance_normalized(MT1[keypoint_middle_left], MT2[keypoint_middle_right])
        dis_left_right_ring = dtw_distance_normalized(MT1[keypoint_ring_left], MT2[keypoint_ring_right])
        dis_left_right_pinky = dtw_distance_normalized(MT1[keypoint_pinky_left], MT2[keypoint_pinky_right])
        dis_left_right_wrist = dtw_distance_normalized(MT1[keypoint_wrist_left], MT2[keypoint_wrist_right])
        dis_right_left_thumb = dtw_distance_normalized(MT1[keypoint_thumb_right], MT2[keypoint_thumb_left])
        dis_right_left_index = dtw_distance_normalized(MT1[keypoint_index_right], MT2[keypoint_index_left])
        dis_right_left_middle = dtw_distance_normalized(MT1[keypoint_middle_right], MT2[keypoint_middle_left])
        dis_right_left_ring = dtw_distance_normalized(MT1[keypoint_ring_right], MT2[keypoint_ring_left])
        dis_right_left_pinky = dtw_distance_normalized(MT1[keypoint_pinky_right], MT2[keypoint_pinky_left])
        dis_right_left_wrist = dtw_distance_normalized(MT1[keypoint_wrist_right], MT2[keypoint_wrist_left])

        ## calculate the mean distance for the opposite hands
        distance_left_right = np.mean([dis_left_right_thumb, dis_left_right_index, dis_left_right_middle,
                                        dis_left_right_ring, dis_left_right_pinky, dis_left_right_wrist])
        distance_right_left = np.mean([dis_right_left_thumb, dis_right_left_index, dis_right_left_middle,
                                        dis_right_left_ring, dis_right_left_pinky, dis_right_left_wrist])


    ### calculate the distance for flipped videos
    if "_" in hands_dtw and use_mirrored_ts: # if hands_dtw is left_right or right_left and use_mirrored_ts is True
        if reference == "A":
            MT2 = MT_B_flipped
        else:
            MT1 = MT_B_flipped
        
        ### left hand ###
        ### xyz
        dis_left_thumb = dtw_distance_normalized(MT1[keypoint_thumb_left], MT2[keypoint_thumb_left])
        dis_left_index = dtw_distance_normalized(MT1[keypoint_index_left], MT2[keypoint_index_left])
        dis_left_middle = dtw_distance_normalized(MT1[keypoint_middle_left], MT2[keypoint_middle_left])
        dis_left_ring = dtw_distance_normalized(MT1[keypoint_ring_left], MT2[keypoint_ring_left])
        dis_left_pinky = dtw_distance_normalized(MT1[keypoint_pinky_left], MT2[keypoint_pinky_left])
        dis_left_wrist = dtw_distance_normalized(MT1[keypoint_wrist_left], MT2[keypoint_wrist_left])
        ## calculate the mean distance for the hands
        distance_left_mirrored = np.mean([dis_left_thumb, dis_left_index, dis_left_middle, 
                                            dis_left_ring, dis_left_pinky, dis_left_wrist])

        ### right hand ###
        ### xyz
        dis_right_thumb = dtw_distance_normalized(MT1[keypoint_thumb_right], MT2[keypoint_thumb_right])
        dis_right_index = dtw_distance_normalized(MT1[keypoint_index_right], MT2[keypoint_index_right])
        dis_right_middle = dtw_distance_normalized(MT1[keypoint_middle_right], MT2[keypoint_middle_right])
        dis_right_ring = dtw_distance_normalized(MT1[keypoint_ring_right], MT2[keypoint_ring_right])
        dis_right_pinky = dtw_distance_normalized(MT1[keypoint_pinky_right], MT2[keypoint_pinky_right])
        dis_right_wrist = dtw_distance_normalized(MT1[keypoint_wrist_right], MT2[keypoint_wrist_right])
        ## calculate the mean distance for the hands
        distance_right_mirrored = np.mean([dis_right_thumb, dis_right_index, dis_right_middle, 
                                            dis_right_ring, dis_right_pinky, dis_right_wrist])


    if hands_dtw == "both":
        distance = np.mean([distance_left, distance_right])
    elif hands_dtw == "left":
        distance = distance_left
    elif hands_dtw == "right":
        distance = distance_right
    elif "_" in hands_dtw:
        if reference == "A": # when speaker A is MT1 and speaker B MT2
            if hands_dtw == "left_right": # speaker A produces left-hand gesture and speaker B produces right-hand gesture
                if use_mirrored_ts:
                    distance = min(distance_left_right, distance_left_mirrored)
                else:
                    distance = distance_left_right
            if hands_dtw == "right_left": # speaker A produces right-hand gesture and speaker B produces left-hand gesture
                if use_mirrored_ts:
                    distance = min(distance_right_left, distance_right_mirrored)
                else:
                    distance = distance_right_left
        elif reference == "B": # when speaker B is MT1 and speaker A MT2
            if hands_dtw == "left_right": # speaker B produces left-hand gesture and speaker A produces right-hand gesture
                if use_mirrored_ts:
                    distance = min(distance_right_left, distance_left_mirrored)
                else:
                    distance = distance_right_left
            if hands_dtw == "right_left":
                if use_mirrored_ts:
                    distance = min(distance_left_right, distance_right_mirrored)
                else:
                    distance = distance_left_right

    #make a np array and append the each distance to the dataframe
    distance_array = np.append(distance_array, 
                               [distance,
                               ## xyz
                                dis_left_thumb, dis_right_thumb, dis_left_index, dis_right_index, 
                                dis_left_middle, dis_right_middle, dis_left_ring, dis_right_ring, 
                                dis_left_pinky, dis_right_pinky, dis_left_wrist, dis_right_wrist,
                                dis_left_right_thumb, dis_left_right_index, dis_left_right_middle,
                                dis_left_right_ring, dis_left_right_pinky, dis_left_right_wrist,
                                dis_right_left_thumb, dis_right_left_index, dis_right_left_middle,
                                dis_right_left_ring, dis_right_left_pinky, dis_right_left_wrist])

    return distance_array


def make_export_dtw_df(dtw_folder, ts_annot_folder, keypoints, anno, use_mirrored_ts=False):
    # make an empty dataframe to store the results
    df_distance = pd.DataFrame(columns=["pair", "comparison_id", "average_distance",
                                        ## xyz
                                        "left_thumb", "right_thumb", "left_index", "right_index", 
                                        "left_middle", "right_middle", "left_ring", "right_ring", 
                                        "left_pinky", "right_pinky", "left_wrist", "right_wrist",
                                        "left_right_thumb", "left_right_index", "left_right_middle",
                                        "left_right_ring", "left_right_pinky", "left_right_wrist",
                                        "right_left_thumb", "right_left_index", "right_left_middle",
                                        "right_left_ring", "right_left_pinky", "right_left_wrist"])

    # specify columns we want to keep in the timeseries dataframe (before merging with annotations)
    cols_to_keep = ["File", "Speaker", "comparison_id", "time"]
    cols_to_keep.extend(keypoints)

    error_count = 0
    error_files = []

    # using dtw to compare distances and show a warping line
    dtw_output = dtw_folder + "dtw_distance_mirrored.csv" if use_mirrored_ts else dtw_folder + "dtw_distance.csv"
    if os.path.exists(dtw_output):
        print("DTW distance file already exists.")
    else:
        ts_annot_folder_files = [file for file in os.listdir(ts_annot_folder) if file.endswith(".csv")]
        for filename in tqdm(ts_annot_folder_files):
            pair = filename.split("_")[0]
            speaker = filename.split("_")[1].upper()
            comparison_id = filename.split("_")[2]
            hands_dtw = anno[anno["comparison_id"] == int(comparison_id)]["hands_dtw"].values[0]
            distance_array = np.array([pair, comparison_id])

            if speaker == "B":
                continue
            else:
                MT_A = pd.read_csv(ts_annot_folder + filename)
                MT_B = pd.read_csv(ts_annot_folder + filename.replace("a", "b"))
                MT_B_flipped = np.nan
                if "_" in hands_dtw and use_mirrored_ts: # if hands_dtw is left_right or right_left and use_mirrored_ts is True
                    MT_B_flipped = pd.read_csv(ts_annot_folder + "flipped/" + filename.replace("a", "b"))
                    MT_B_flipped = MT_B_flipped[cols_to_keep]
                    

                ### make the earlier gesture as the referent gesture (makes a difference for asymmetric DTW)
                if MT_A.loc[0, "time"] < MT_B.loc[0, "time"]:
                    reference = "A"
                    MT1 = MT_A[cols_to_keep]
                    MT2 = MT_B[cols_to_keep]
                else:
                    reference = "B"
                    MT1 = MT_B[cols_to_keep]
                    MT2 = MT_A[cols_to_keep]

                ### compute DTW distance
                try:
                    distance_array = dtw_distance_dependent(MT1, MT2, MT_B_flipped, reference, distance_array, 
                                                            hands_dtw, use_mirrored_ts, keypoints)
                    df_distance = pd.concat([df_distance, pd.DataFrame([distance_array], columns=df_distance.columns)])
                except:
                    error_count += 1
                    error_files.append(filename)
                    # print(f"[DTW]Error in: {filename}. The file might contain missing values for the keypoints or have too few datapoints. Skipping this file...")
                    pass # do nothing and continue to the next line


        # sort the dataframe by comparison_id
        df_distance["comparison_id"] = df_distance["comparison_id"].astype(int)
        df_distance = df_distance.sort_values(by=["comparison_id"])

        anno = anno.drop(columns=["pair"])
        df_distance = pd.merge(df_distance, anno, on="comparison_id", how="left")

        # save the dataframe to a csv file
        df_distance.to_csv(dtw_output, index=False)

        # check the shape of the dataframe
        print(f"The follwing {error_count} files were skipped. The files might contain missing values for the keypoints or have too few datapoints.")
        print(error_files)
        print(df_distance.shape)
    
