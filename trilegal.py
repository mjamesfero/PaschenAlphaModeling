import selenium
from selenium.webdriver import Firefox
from selenium.webdriver.support.ui import Select
import time
import datetime
import os

def get_trilegal(coord_sys, l_coord, b_coord, area, catalog, fil_num, mag_max):
	#'''
	#The purpose of this is to get data from TRILEGAL automatically
	#It is still in its infancy.
	#'''
	#using firefox
	driver = Firefox()
	#website
	driver.get('http://stev.oapd.inaf.it/cgi-bin/trilegal')
	#pause to look human (for now)
	
	#locate desired fields
	#coord_type = driver.find_element_by_name('gal_coord')
	coord_l = driver.find_element_by_name('gc_l')
	coord_b = driver.find_element_by_name('gc_b')
	coord_a = driver.find_element_by_name('eq_alpha')
	coord_d = driver.find_element_by_name('eq_delta')
	field = driver.find_element_by_name('field')
	sys_type = Select(driver.find_element_by_name('photsys_file'))
	filter_type = driver.find_element_by_name('icm_lim')
	limit_mag = driver.find_element_by_name('mag_lim')

	#inputting info
	if coord_sys.lower()[0] == 'g':
		#coord_type = driver.find_element_by_partial_link_text('Galactic coordinates')
		coord_l.clear()
		coord_b.clear()
		coord_l.send_keys(l_coord)
		coord_b.send_keys(b_coord)
	else:
		#coord_type = driver.find_element_by_partial_link_text('Equatorial coordinates')
		coord_a.clear()
		coord_d.clear()
		coord_a.send_keys(l_coord)
		coord_d.send_keys(b_coord)
	#coord_type.click()

	field.clear()
	filter_type.clear()
	limit_mag.clear()
	field.send_keys(area)
	filter_type.send_keys(fil_num)
	limit_mag.send_keys(mag_max)
	
	phot_systems = {'2mass spitzer': '2mass_spitzer.dat',
					'2mass wise': '2mass_spitzer_wise.dat',
					'2mass jhk': '2mass.dat',
					'vista': 'vista.dat',}
	value = 'tab_mag_odfnew/tab_mag_'+phot_systems[catalog]
	sys_type.select_by_value(value)
	driver.find_element_by_name('submit_form').click()

	results = driver.find_element_by_link_text('THIS LINK')

	time.sleep(130)
	results.click()
	
get_trilegal('galactic', 0, 0, 1, '2mass jhk', 3, 6)