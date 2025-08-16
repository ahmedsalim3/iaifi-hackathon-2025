The Galaxy Zoo team regularly receives requests for subject images for various versions of Galaxy Zoo, in order to facilitate other investigations, e.g. machine learning projects. This repository is an updated attempt to provide those in a way that is useful to the wider community.

The images here are meant to be used with the data tables available at data.galaxyzoo.org. They are the "original" sample of subject images in Galaxy Zoo 2 (Willett et al. 2013, MNRAS, 435, 2835, DOI: 10.1093/mnras/stt1458) as identified in Table 1 of Willett et al. and also in Hart et al. (2016, MNRAS, 461, 3663, DOI: 10.1093/mnras/stw1588). The original GZ2 subjects also gave the option to view an inverted version of the subject image; these inverted images are not provided but are easily reproducible from the included subject images. 

If you use this dataset, please cite Willett et al. (2013) as the general data release and also cite the DOI for this dataset; if you use the updated debiased tables from Hart et al. (2016) please cite that as well.

There are 243,434 images in total. This is off by about 0.08% from the total count in the tables - it's not clear what the cause of the discrepancy is, but we don't think the missing images have any particular sampling bias, so this sample should be useful for research.

The images are available in a single zip file (images_gz2.zip).

The most recent and reliable source for morphology measurements is "GZ2 - Table 1 - Normal-depth sample with new debiasing method – CSV" (from Hart et al. 2016), which is available at https://data.galaxyzoo.org. To cross-reference the images with Table 1, this sample includes another CSV table (gz2_filename_mapping.csv) which contains three columns and 355,990 rows. The columns are:

- objid: the Data Release 7 (DR7) object ID for each galaxy. This should match the first column in Table 1.
- sample: string indicating the subsampling of the galaxy.  
- asset_id: an integer that corresponds to the filename of the image in the zipped file linked above.

As an example row:

587722981742084144,original,16

The galaxy is 587722981741363294, which is in Table 1 and was identified by GZ2 volunteers as a barred spiral galaxy with a mild bulge and two tightly-wound arms (morphology='Sc2t'). It is in the original GZ2 sample, and can be found in the zipped file as 16.jpg. 

The overlap between the set of images, the attached table, and Table 1 is not 100%; there are a few rows in the tables that don't have a corresponding image. Again, it's not clear what the exact reason is for this, but we suggest just dropping any missing rows/images from your analysis unless you have a need for analyzing specific subjects. If you do need a 100% complete sample, you can obtain the missing images directly from SDSS. 

Based on spot checks the mappings between asset ID and DR7 object ID appear correct, but we strongly suggest that you pick some random images and verify on your own that the image seems to match the label/classifications that are listed in Table 1. 

If you have any issues using this dataset, please contact the Galaxy Zoo team, in particular Brooke Simmons (b.simmons@lancaster.ac.uk). Should Dr Simmons be unavailable, try contacting Karen Masters or Chris Lintott.

- the GZ team, 5 Dec 2019
