import tarfile
!wget http://www-rech.telecom-lille.fr/shrec2017-hand/HandGestureDataset_SHREC2017.tar.gz
tar = tarfile.open("/content/HandGestureDataset_SHREC2017.tar.gz", "r:gz")
tar.extractall("/content/")
