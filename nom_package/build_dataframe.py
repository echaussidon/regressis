import logging
logger = logging.getLogger("build_dataframe")

import sys, os
# to avoid error from pandas method into the logger -> pandas use NUMEXPR Package
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import numpy as np
import healpy as hp
import fitsio
import pandas as pd

from desi_footprint import DR9_footprint

import matplotlib.pyplot as plt

#from desitarget.geomask import hp_in_box ## --> on va copier la fonction pour eviter une dependance


## mettre commentaire cii
zone_name_to_column_name = {'North':'ISNORTH', 'South':'ISSOUTHWITHOUTDES', 'Des':'ISDES',
                            'South_mid':'ISSOUTHMID', 'South_pole':'ISSOUTHPOLE',
                            'Des_mid':'ISDESMID', 'South_all':'ISSOUTH'}


class DataFrame(object):
    """
    Build the dataframe needed to compute the weights due to photometry / spectroscopy systematic effects
    """
    def __init__(self, version, tracer, data_dir, output_dir, suffixe_tracer='', **kwargs):
        """
        Initialize :class:`DataFrame`

        Parameters
        ----------

        """
        self.version = version
        self.tracer = tracer
        self.suffixe_tracer = suffixe_tracer

        self.data_dir = data_dir #la ou les maps sont sauvagarder (plus tard aussi les targets d'ashley)
        self.output_dir = output_dir # la ou on va sauvegarder le resulat et le dessin --> simplifier le nom

        self.Nside = kwargs.get('Nside', None)
        # info to normalize the target density
        self.use_median = kwargs.get('use_median', False)
        self.use_new_norm = kwargs.get('use_new_norm', False)

        # footprint info
        self.remove_LMC = kwargs.get('remove_LMC', False)
        self.clean_south = kwargs.get('clean_south', True)
        self.cut_DESI = kwargs.get('cut_DESI', False)

        # which region we want to use --> if None use default
        self.region = kwargs.get('region', None)

        if self.region is None:
            logger.info('Use default region')
            self.region = ['North', 'South', 'Des']
        logger.info(f"REGION: {self.region}")

        self.output = os.path.join(output_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}')
        if os.path.isdir(self.output):
            logger.info('Ouput folder already exists')
        else:
            logger.info(f'Create output directory: {self.output}')
            os.mkdir(self.output)
            os.mkdir(os.path.join(self.output, 'Build_dataFrame'))


    def load_feature_from_healpix(self, data_dir=None):
        """
        Load photometric templates and footprint info

        Parameters
        ----------
        data_dir: str
            If None use self.data_dir as specific data directory otherwise use data_dir to load the photometric templates
        """
        if data_dir is None:
            data_dir = self.data_dir

        # Load DR9 Legacy Imaging footprint
        DR9 = DR9_footprint(self.Nside, remove_LMC=self.remove_LMC, clear_south=self.clean_south)
        footprint = DR9.load_footprint()
        if self.cut_DESI: #restricted to DESI footprint
            logger.info('Restrict footrpint to DESI footprint')
            footprint[hp_in_box(self.Nside, [0, 360, -90, -30])] = False

        # extract the different region from DR9_footprint
        north, south, des = DR9.load_photometry(remove_around_des=True)
        _, south_mid, south_pole = DR9.load_elg_region()
        des_mid = des & ~south_pole
        photometry_footprint = pd.DataFrame({'FOOTPRINT': footprint, 'ISNORTH':north, 'ISSOUTH':south, 'ISDES':des, 'ISSOUTHWITHOUTDES':south&~des, 'ISSOUTHMID':south_mid, 'ISSOUTHPOLE':south_pole, 'ISDESMID':des_mid})

        # Load pixmap from desitarget
        logger.info(f"Read {os.path.join(self.data_dir, f'pixweight-dr9-{self.Nside}.fits')}")
        sel_columns = ['HPXPIXEL', 'FRACAREA_12290', 'STARDENS', 'EBV',
                       'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                       'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z',
                       'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z']
        feature_pixmap = pd.DataFrame(fitsio.FITS(os.path.join(self.data_dir, f'pixweight-dr9-{self.Nside}.fits'))[1][sel_columns].read().byteswap().newbyteorder())
        self.fracarea = feature_pixmap['FRACAREA_12290']
        # Ok pour l'isntant --> pour les targets de desi c'est FRACAREA --> faire quelqe chose si on veut load un external fracarea comme pour DA02

        # Load Sgr. Stream map
        logger.info(f"Read {os.path.join(self.data_dir, f'sagittarius_stream_{self.Nside}.npy')}")
        stream_map = pd.DataFrame(np.load(os.path.join(self.data_dir, f"sagittarius_stream_{self.Nside}.npy")), columns=['STREAM'])
        self.pixmap = pd.concat([stream_map, feature_pixmap, photometry_footprint], axis=1)


    def load_target_from_healpix(self, data_dir=None, load_fracarea=False):
        """
        Load precomputed healpix target density map and specific fracarea if require

        Parameters
        ----------
        data_dir: str
            If None use self.data_dir as specific data directory otherwise use data_dir to load the target density map
        load_fracarea: bool
            If False use the default value of the fracarea (FRACAREA_12290) otherwise load _fracrea file in the same folder than the density map
        """
        if data_dir is None:
            data_dir = self.data_dir
        logger.info(f"Read {os.path.join(data_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.npy')}")
        self.targets = np.load(os.path.join(data_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.npy'))

        if load_fracarea:
            logger.info(f"Read {os.path.join(data_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_fracarea_{self.Nside}.npy')}")
            self.targets = np.load(os.path.join(data_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_fracarea_{self.Nside}.npy'))

    def load_target_from_catalog(self, data_dir=None):
        """
        Load precomputed healpix target density map

        Parameters
        ----------
        data_dir: str
            If None use self.data_dir as specific data directory otherwise use data_dir to load the target density map
        """
        if data_dir is None:
            data_dir = self.data_dir
        logger.info(f"Read {os.path.join(data_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.npy')}\n")
        self.targets = np.load(os.path.join(data_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.npy'))



    def build_dataframe_photometry(self):
        """
        Build/Clean the dataframe and select the training values for the photometry case
        """

        footprint = self.pixmap['FOOTPRINT']
        pixels = self.pixmap['HPXPIXEL'][footprint]
        features = self.pixmap[footprint] # copy is automatic since we extract a sub matric
        fracarea = self.fracarea[footprint]
        targets = self.targets[footprint]

        # remove pixels with wrong value of targets and with 0 fracarea ...
        # pixels at the border of the footprint have NaN value and are not completly remove when i take footrpint (ok no problemo here)+ pixels that we mask in function of suffixe :)
        # this works also when we want to remove specific area setting fracara 0  or np.nan for targets somewhere --> cf DA02
        pix_to_remove = np.isnan(targets)
        #mettre aussi np.isnan pour fracarea
        print("\n IL FAUT FAIRE BIEN LE LOGGER LA -> il faut affihcer des infos clairs et le reste le metter en commentaire au cas ou il faut comprendre")
        print("\nNombre de pixel qui ont une fraction de leur surface observee nulle : {}".format(np.sum(fracarea == 0)))
        print("Nombre de ces pixels qui ne font pas partie des pixels a retirer definie plus haut : {}".format(np.sum((fracarea == 0) & ~(pix_to_remove))))
        print("OK --> ceux sont des pixels du bords pas de soucis")
        pix_to_remove |= ~(fracarea > 0)

        pixels = pixels[~pix_to_remove]
        fracarea = fracarea[~pix_to_remove]
        features = features[~pix_to_remove]
        targets = targets[~pix_to_remove]

        #pixels with bad behaviour --> reject for training (cf version notebook pour les histogrames et tout)
        keep_to_train = np.ones(targets.size)

        print("\n mettre un logger correct pour expliquer [ICI] \n")
        print("Faire un beau dessin aussi pour qu'on puisse erte ok")
        if self.Nside == 512:
            pix_to_remove_train = (fracarea < 0.5) | (fracarea > 1.5)
            pix_to_remove_train &= (targets < np.quantile(targets, 0.1)) | (targets < np.quantile(targets, 0.9))
        elif self.Nside == 256:
            pix_to_remove_train = (fracarea < 0.9) | (fracarea > 1.1)
            pix_to_remove_train &= (targets < np.quantile(targets, 0.1)) | (targets < np.quantile(targets, 0.9))

        elif version == 'DA02':
            pix_to_remove_train = ~(fracarea > 0)
            print("[INFO] FOR DA02 on a deja fait la selection lorsque l'on cree les pixmap")

        keep_to_train[pix_to_remove_train] = 0

        print("\n/!\ I y a {} pixel avec des proprietes anormales pour l'entrainement ce qui represente {:2.2%} du footprint de DESI".format(np.sum(pix_to_remove_train), np.sum(pix_to_remove_train)/pix_to_remove_train.size))

        plt.figure(figsize=(8,6))
        plt.hist(targets, range=(0,100), bins=100)
        plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"test_remove_targets_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))
        plt.close()

        plt.figure(figsize=(8,6))
        plt.hist(fracarea, range=(0.5, 1.4), bins=100)
        plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"test_remove_fracarea_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))
        plt.close()

        # On regarde qu'elles sont les pixels qui ont un problème
        tmp = np.zeros(hp.nside2npix(self.Nside))
        tmp[pixels[pix_to_remove_train == True]] = 1
        plt.figure(figsize=(8,6))
        hp.mollview(tmp, rot=120, nest=True, title='strange pixel', cmap='jet')
        plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"strange_pixel_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))

        # build normalized targets
        normalized_targets = np.zeros(targets.size)

        for zone_name in self.region:
            pix_zone = features[zone_name_to_column_name[zone_name]].values
            pix_to_use = pix_zone & (keep_to_train == 1)

            # only conserve pixel in the correct radec box
            if self.use_new_norm:
                #compute normalization on subpart of the footprint which is not contaminated for the north and the south !
                keep_to_norm = np.zeros(hp.nside2npix(Nside))
                if zone_name == 'North':
                    keep_to_norm[hp_in_box(Nside, [120, 240, 32.2, 40], inclusive=True)] = 1
                elif zone_name == 'South':
                    keep_to_norm[hp_in_box(Nside, [120, 240, 24, 32.2], inclusive=True)] = 1
                else:
                    keep_to_norm = np.ones(hp.nside2npix(Nside))
                keep_to_norm = keep_to_norm[footprint][~pix_to_remove]
                pix_to_use &= keep_to_norm

            # compute the mean only on pixel with "correct" behaviour
            if not self.use_median:
                mean_targets_density_estimators = np.mean(targets[pix_to_use] / fracarea[pix_to_use])
            else:
                mean_targets_density_estimators = np.median(targets[pix_to_use] / fracarea[pix_to_use])

            #compute normalized_targets every where but we don't care we only use keep_to_train == 1 during the training
            normalized_targets[pix_zone] = targets[pix_zone] / (fracarea[pix_zone]*mean_targets_density_estimators)

            print("\n  ** INFO for ", zone_name)
            print(mean_targets_density_estimators)
            print(normalized_targets[pix_to_use].mean())

        plt.figure(figsize=(8,6))
        plt.hist(normalized_targets[keep_to_train == 1], range=(0,5), bins=100)
        plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"normalized_targets_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))

        print("\nNombre de case 'nan' dans les features : ", features.isnull().sum().sum(), ' --> OK ON VA REMPLIR AVEC LES MOYENNES ...')
        print("Nombre de lignes (ie) de pixels qui posent probleme dans features : ", features[features.isnull().T.any().T].index.size)
        print("Nombre de case 'nan' dans la densite : ", np.isnan(normalized_targets).sum(), ' --> OUF !')
        print("Nombre de case 'nan' dans les pixels : ", pixels.isnull().sum(), ' -- > OUF !')

        features = features.fillna(features.mean())

        print("Nombre de lignes (ie) de pixels qui posent probleme apres remplissage : ", features[features.isnull().T.any().T].index.size)
        print("\n \n \n ")
        # Attention pixels est déjà dans features
        self.data_regressor = pd.concat([features,
                                          pd.DataFrame(normalized_targets, index=pixels, columns=['NORMALIZED_TARGETS']),
                                          pd.DataFrame(keep_to_train, index=pixels, columns=['KEEP_TO_TRAIN'])], axis=1)


    def build_dataframe_global():
        """
        Build the dataframe and select the training values for the photometry/spectroscopy case
        """

        return 1
