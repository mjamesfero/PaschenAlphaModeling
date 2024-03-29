import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from astropy.io import ascii
import numpy as np
import pandas 
import csv 
import glob

class Trilegal:

    def __init__(self, Equitorial=False, phot_system='2mass jhk'):
        #this translates between galactic coordinates and equitorial coordinates
        if Equitorial:
            coord = '2'
        else:
            coord = '1'
        #this determines the photometry system used
        #I had this be included in the initialization because it seemed important
        self.phot_syst = phot_system
        phot_systems = {'2mass spitzer': 'tab_mag_2mass_spitzer.dat',
					'2mass wise': 'tab_mag_2mass_spitzer_wise.dat',
					'2mass jhk': 'tab_mag_2mass.dat',
                    'ogle spitzer': 'tab_mag_ogle_2mass_spitzer.dat',
                    'ubvrijhk': 'tab_mag_ubvrijhk.dat',
                    'ubvrijhklmn': 'tab_mag_bessell.dat',
                    'akari': 'tab_mag_akari.dat',
                    'batc': 'tab_mag_batc.dat',
                    'ukidss': 'tab_mag_ukidss.dat',
                    'uvit': 'tab_mag_uvit.dat',
                    'visir': 'tab_mag_visir.dat',
					'vista': 'tab_mag_vista.dat',
                    'vphas': 'tab_mag_vphas.dat',
                    'vst': 'tab_mag_vst_omegacam.dat',
                    'vilnius': 'tab_mag_vilnius.dat',
                    'wfc3': 'tab_mag_wfc3_uvisCaHK.dat',
                    'wfirst': 'tab_mag_wfirst_proposed2017.dat',
                    'washington': 'tab_mag_washington_ddo51.dat',
                    'deltaa': 'tab_mag_deltaa.dat'}
        phot_index = phot_system.lower()
        photsys = phot_systems[phot_index]
        #this is all the data sent
        self.data = {'submit_form': 'Submit',
        'trilegal_version': '1.6',
        'gal_coord': coord,
        'gc_l': '0',
        'gc_b': '0.1',
        'eq_alpha': '0',
        'eq_delta': '0',
        'field': '1',
        'photsys_file': f'tab_mag_odfnew/{photsys}',
        'icm_lim': '4',
        'mag_lim': '26',
        'mag_res': '0.1',
        'imf_file': 'tab_imf/imf_chabrier_lognormal.dat',
        'binary_kind': '1',
        'binary_frac': '0.3',
        'binary_mrinf': '0.7',
        'binary_mrsup': '1',
        'output_printbinaries': '0',
        'extinction_h_z': '110',
        'extinction_h_r': '100000',
        'extinction_rho_sun': '0.002',
        'extinction_kind': '1',
        'extinction_infty': '0.0378',
        'extinction_sigma': '0',
        'r_sun': '8700',
        'z_sun': '24.2',
        'thindisk_kind': '3',
        'thindisk_h_z0': '94.6902',
        'thindisk_hz_tau0': '5.55079e9',
        'thindisk_hz_alpha': '1.6666',
        'thindisk_h_r': '2913.36',
        'thindisk_r_min': '0',
        'thindisk_r_max': '15000',
        'thindisk_rho_sun': '55.4082',
        'thindisk_file': 'tab_sfr/file_sfr_thindisk_mod.dat',
        'thindisk_a': '0.735097',
        'thindisk_b': '0',
        'thickdisk_kind': '3',
        'thickdisk_h_z': '800',
        'thickdisk_h_r': '2394.07',
        'thickdisk_r_min': '0',
        'thickdisk_r_max': '15000',
        'thickdisk_rho_sun': '0.0010',
        'thickdisk_file': 'tab_sfr/file_sfr_thickdisk.dat',
        'thickdisk_a': '1',
        'thickdisk_b': '0',
        'halo_kind': '2',
        'halo_r_eff': '2698.93',
        'halo_q': '0.583063',
        'halo_rho_sun': '0.000100397',
        'halo_file': 'tab_sfr/file_sfr_halo.dat',
        'halo_a': '1',
        'halo_b': '0',
        'bulge_kind': '2',
        'bulge_am': '2500',
        'bulge_a0': '95',
        'bulge_eta': '0.68',
        'bulge_csi': '0.31',
        'bulge_phi0': '15',
        'bulge_rho_central': '406.0',
        'bulge_file': 'tab_sfr/file_sfr_bulge_zoccali_p03.dat',
        'bulge_a': '1',
        'bulge_b': '-2.0e9',
        'object_kind': '0',
        'object_mass': '1280',
        'object_dist': '1658',
        'object_av': '1.504',
        'object_avkind': '1',
        'object_file': 'tab_sfr/file_sfr_m4.dat',
        'object_a': '1',
        'object_b': '0',
        'output_kind': '1',
        'object_kind': '0',
        'bulge_cutoffmass': '0.01',
        'object_cutoffmass': '0.8',
        '.cgifields': 'gal_coord',
        '.cgifields': 'extinction_kind',
        '.cgifields': 'binary_kind',
        '.cgifields': 'output_printbinaries',
        '.cgifields': 'bulge_kind',
        '.cgifields': 'thindisk_kind',
        '.cgifields': 'halo_kind',
        '.cgifields': 'thickdisk_kind'}
    
    def _punctuation(self, number):
        """
        Turns normal strings into computer friendly file names.
        The notation is 'p' stands for period and 'n' stands for negative.
        Spaces are replaced with underscores.
        """
        formatted_number = ''
        word = number
        if number is not str:
            word = str(number)
        for letter in word:
            if letter is '.':
                formatted_number += 'p'
            elif letter is '-':
                formatted_number += 'n'
            elif letter is ' ':
                formatted_number += '_'
            else:
                formatted_number += letter
        
        return formatted_number

    def search(self, gc_l=0, gc_b=0.1, field=0.001, icm_lim=3, mag_lim=16):
        """
        Searches a given field at a given lon and lat. The filter and magnitude limit
        are also set here. The query is run through TRILEGAL, and if there are results,
        they are saved to a .csv file.

        Parameters
        ----------
        gc_l : float
                The longitude being searched.

        gc_b : float
                The latitude being searched.

        field : float
                The size of the field in deg^2.

        icm_lim : int
                The limiting filter, set on 3 as default.
        
        mag_lim : int
                The upper limit of the (icm_lim)th filter. 

        Returns
        -------
        datatable : ascii table and .dat file
                Results from TRILEGAL query (or 'No stars found.' if no stars were found)
                stored as an ascii table and saved as a .dat file using the naming sequence
                '{coor}_{glon}_{glat}_{gfield}_{system_name}_{save_time}.csv'. If no stars were
                found, no .dat file is saved.

        """
        if self.data['gal_coord'] == '1':
            coor = 'gal'
            self.data['gc_l'] = str(gc_l)
            self.data['gc_b'] = str(gc_b)
        elif self.data['gal_coord'] == '2':
            coor='fk5'
            self.data['eq_alpha'] = str(gc_l)
            self.data['eq_delta'] = str(gc_b)
        self.data['field'] = str(field)
        self.data['icm_lim'] = str(icm_lim)
        self.data['mag_lim'] = str(mag_lim)

        glon = self._punctuation(gc_l)
        glat = self._punctuation(gc_b)
        gfield = self._punctuation(field)
        system_name = self._punctuation(self.phot_syst)

        response = requests.post("http://stev.oapd.inaf.it/cgi-bin/trilegal_1.6", data=self.data)
        soup = BeautifulSoup(response.text, features="html5lib")
        url = soup.find('input', attrs={'name':'outurl'}).attrs['value']
        newurl = urljoin(response.url, url)

        while 1:
            datatable_response = requests.get(newurl)
            if "#TRILEGAL normally terminated" in datatable_response.text:
                break
            time.sleep(5)

        try:
            import pdb; pdb.set_trace()
            text = datatable_response.text.replace('\n#TRILEGAL normally terminated\n','')
            datatable = ascii.read(text)
            save_time = time.time()
            file_name = f"{coor}_{glon}_{glat}_{gfield}_{system_name}_{save_time}.csv"
            ascii.write(datatable, file_name, overwrite=True)
        except:
            datatable = 'No stars found.'
        return datatable
    
    def search_arcmin(self, field=50, gc_l=0, gc_b=90, icm_lim=3, mag_lim=16):
        """
        Same as search() except field is in arcminutes.
        """
        field_arcmin = field/3600
        results = self.search(field=field_arcmin, gc_l=0, gc_b=90, icm_lim=3, mag_lim=18)
        return results
    
    def search_arcsec(self, field=50, gc_l=0, gc_b=90, icm_lim=3, mag_lim=16):
        """
        Same as search() except field is in arcseconds.
        """
        field_arcsec = field/(3600*3600)
        results = self.search(field=field_arcsec, gc_l=0, gc_b=90, icm_lim=3, mag_lim=18)
        return results
    
    def extinction_params(self, kind=1, h_z=110, h_r=100000, rho_sun=0.002, infinity=0.0378, sigma=0):
        """
        For 'no dust extinction or exponential disk exp(-|z|/hz,dust)×exp(-R/hR,dust) 
        with scale heigth hz,dust= pc and scale length hR,dust= pc' type 'kind = 0'.
        For local calibration, type 'kind = 1'.
        For calibration at infinity, type 'kind=2'.
        """
        self.data['extinction_h_z'] = str(h_z)
        self.data['extinction_h_r'] = str(h_r)
        self.data['extinction_rho_sun'] = str(rho_sun)
        self.data['extinction_kind'] = str(kind)
        self.data['extinction_infty'] = str(infinity)
        self.data['extinction_sigma'] = str(sigma)
        return
    
    def _try_to_get(self,gc_l=0, gc_b=0.1, field=0.001):
        """
        Internal function for returning the most recently generated file
        for a given region and catalog. If none exists, it will move on to
        searching the region as usual.
        """
        if self.data['gal_coord'] == '1':
            coor = 'gal'
        elif self.data['gal_coord'] == '2':
            coor='fk5'
        glon = self._punctuation(gc_l)
        glat = self._punctuation(gc_b)
        gfield = self._punctuation(field)
        system_name = self._punctuation(self.phot_syst)
        filename = f"{coor}_{glon}_{glat}_{gfield}_{system_name}_*.csv"
        files = glob.glob(filename)
        print(files)
        if files:
            fn=files[-1]
            print(fn)
            file = pandas.read_csv(fn)
            return file
        else:
            file = False
            return file
        
    def big_search(self, gc_l=0, gc_b=0.1, field=0.001, icm_lim=3, mag_lim=16):
        """
        A modified version of self.search that allows for jobs to be done in batches 
        for large data sets. 
        Pulls up previous searches to avoid unnecessary computing time.
        """
        file = self._try_to_get(gc_l=gc_l, gc_b=gc_b, field=field)
        if not file.empty:
            return file
        else:
            new_file = self.search(gc_l=gc_l, gc_b=gc_b, field=field, icm_lim=icm_lim, mag_lim=mag_lim)
            return new_file


# field = (47.5/3600)**2
# test = Trilegal()
# test_res = test.search(field=field)
# print(test_res)
