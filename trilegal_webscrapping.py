import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
from astropy.io import ascii

class Trilegal:

    def __init__(self, coord_sys='g', phot_system='2mass jhk'):
        if coord_sys.lower()[0] is 'g':
            coord = '1'
        elif coord_sys.lower()[0] is 'e':
            coord = '2'
        else:
            coord = '1'
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
        self.data = {'submit_form': 'Submit',
        'trilegal_version': '1.6',
        'gal_coord': coord,
        'gc_l': '0',
        'gc_b': '90',
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
        'extinction_rho_sun': '0.00015',
        'extinction_kind': '2',
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
        formatted_number = ''
        word = number
        if number is not str:
            word = str(number)
        for letter in word:
            if letter is '.':
                formatted_number += 'p'
            elif letter is ' ':
                formatted_number += '_'
            else:
                formatted_number += letter
        
        return formatted_number

    def search(self, gc_l=0, gc_b=90, field=0.001):
        self.data['gc_l'] = str(gc_l)
        self.data['gc_b'] = str(gc_b)
        self.data['field'] = str(field)

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
            datatable = ascii.read(datatable_response.text)
            save_time = time.time()
            file_name = f"gal_{glon}_{glat}_{gfield}_{system_name}_{save_time}.dat"
            ascii.write(datatable, file_name, overwrite=True)
        except:
            datatable = 'No stars found.'
        return datatable
    
    def search_arcmin(self, field=50):
        field_arcmin = field/3600
        results = self.search(field=field_arcmin)
        return results
    
    def search_arcsec(self, field=50):
        field_arcsec = field/(3600*3600)
        results = self.search(field=field_arcsec)
        return results


#test = Trilegal(phot_system='vista')
#test_res = test.search_arcmin()
#print(test_res)
#print(test_res['Ks'])
#print(test_res[0][12])