cd /home/waitangwen/Datasets/rgbd_dataset_freiburg3_walking_xyz/Results
evo_ape tum groundtruth.txt KeyFrameTrajectory.txt -v -a --plot 
evo_rpe tum groundtruth.txt CameraTrajectory.txt -va --delta 0.5 --delta_unit m --plot
