Foundations of Astronomical Data Science

Welcome to The Carpentries Etherpad!

This pad is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

Use of this service is restricted to members of The Carpentries community; this is not for general purpose use (for that, try https://etherpad.wikimedia.org).

Users are expected to follow our code of conduct: https://docs.carpentries.org/topic_folders/policies/code-of-conduct.html

All content is publicly available under the Creative Commons Attribution License: https://creativecommons.org/licenses/by/4.0/
 ----------------------------------------------------------------------------  
Course Website: https://abostroem.github.io/2024-01-06-aas/
Lesson Website: https://datacarpentry.org/astronomy-python/07-photo.html
 ----------------------------------------------------------------------------  
Basic Queries: 
  
from astroquery.gaia import Gaia
  
tables = Gaia.load_tables(only_names=True)
  
for table in tables:
    print(table.name)

table_metadata = Gaia.load_table('gaiadr2.gaia_source')
table_metadata
print(table_metadata)

for column in table_metadata.columns:
print(column.name)

ps_metadata = Gaia.load_table("gaiadr2.panstarrs1_original_valid")
print(ps_metadata)
for column in ps_metadata.columns:
print(column.name)

query1 = """
SELECT TOP 10 source_id, ra, dec, parallax
FROM gaiadr2.gaia_source
"""
print(query1)

job1 = Gaia.launch_job(query1)
job1

print(job1)

results1 = job1.get_results()
type(results1)

results1

query2 = """
SELECT TOP 3000 source_id, ra, dec, pmra, pmdec, parallax
FROM gaiadr2.gaia_source
WHERE parallax < 1
"""

job2 = Gaia.launch_job_async(query2)
job2

help(job2)

results2 = job2.get_results()
results2

columns = "source_id, ra, dec, pmra, pmdec, parallax"

query3_base = """
SELECT TOP 10
{my_columns}
FROM gaiadr2.gaia_source
WHERE parallax < 1 AND 
bp_rp BETWEEN -0.75 AND 2
"""

query3 = query3_base.format(my_columns=columns)
print(query3)

job3 = Gaia.launch_job(query3)
job3

results3 = job3.get_results
results3

query3_base = """
SELECT TOP 10
{my_columns}
FROM gaiadr2.gaia_source
WHERE parallax < {max_parallax} AND 
bp_rp BETWEEN -0.75 AND 2
"""

query3 = query3_base.format(my_columns=columns, max_parallax=1)
print(query3)

query3_base = """
SELECT TOP 10
{my_columns}
FROM gaiadr2.gaia_source
WHERE {parallax_criteria} AND 
bp_rp BETWEEN -0.75 AND 2
"""

print(query3_base.format(my_columns=columns, parallax_criteria="parallax between 0.01 and 1"))

Coordinate Transformations:

import astropy.units as u

dir(u)

angle = 10 * u.degree
type(angle)

angle

angle.to(u.arcmin)

angle + 5*u.arcmin

(angle + 5*u.arcmin).to(u.arcmin)

angle + 5*u.kg #will throw error

angle.value

radius = 5 * u.arcmin
radius

radius.to(u.degree)

cone_query = """
SELECT
TOP 10
source_id, ra, dec
FROM gaiadr2.gaia_source
WHERE 1 = CONTAINS(
POINT(ra, dec),
CIRCLE(88.8, 7.4, 0.08333333))
"""

cone_job = Gaia.launch_job(cone_query)
cone_job

cone_results = cone_job.get_results()
cone_results

cone_query = """
SELECT
TOP 10
source_id, ra, dec
FROM gaiadr2.gaia_source
WHERE 1 = CONTAINS(
POINT(ra, dec),
CIRCLE(88.8, 7.4, 0.08333333))
"""

count_query = """
SELECT
count(source_id)
FROM gaiadr2.gaia_source
WHERE CONTAINS(
POINT(ra, dec),
CIRCLE(88.8, 7.4, 0.08333333))=1
"""
count_job = Gaia.launch_job(count_query)
count_results = count_job.get_results()
count_results

def run_query(query):
count_job = Gaia.launch_job(query)
count_results = count_job.get_results()
return count_results

from astropy.coordinates import SkyCoord

# Betelguese
ra = 88.8 * u.degree
dec = 7.4 * u.degree
coord_icrs = SkyCoord(ra=ra, dec=dec, frame='icrs')

coord_icrs

coord_galactic = coord_icrs.transform_to('galactic')
coord_galactic

import gala
dir(gala.coordinates)

from gala.coordinates import GD1Koposov10

gd1_frame = GD1Koposov10()
gd1_frame

coord_gd1 = coord_icrs.transform_to(gd1_frame)
coord_gd1

origin_gd1 = SkyCoord(phi1=0*u.degree, phi2 = 0*u.degree, frame=gd1_frame)

origin_gd1.transform_to('icrs')

phi1_min = -55 * u.degree
phi1_max = -45 * u.degree
phi2_min = -8 * u.degree
phi2_max = 4 * u.degree

def make_rectangle(x1, x2, y1, y2):
"""Return the corners of a rectange"""
xs = [x1, x1, x2, x2, x1]
ys = [y1, y2, y2, y1, y1]
return xs, ys

phi1_rect, phi2_rect = make_rectange(phi1_min, phi1_max, phi2_min, phi2_max)

corners = SkyCoord(phi1=phi1_rect, phi2=phi2_rect, frame=gd1_frame)
print(corners)

corners_icrs = corners.transform_to('icrs')
corners_icrs

corners_list_str = corners_icrs.to_string()
corners_list_str

corners_single_str = ' '.join(corners_list_str)
corners_single_str

corners_single_str.replace(' ', ', ')

def skycoord_to_string(skycoord):
"""Convert a one-dimensional list of SkyCoord to string for Gaia's query format"""
corners_list_str = skycoord.to_string()
corners_single_str = ' '.join(corners_list_str)
return corners_single_str.replace(' ', ', ')

sky_point_list = skycoord_to_string(corners_icrs)
sky_point_list

columns = 'source_id, ra, dec, pmra, pmdec, parallax'

polygon_top10query_base = """
SELECT 
TOP 10
{columns}
FROM gaiadr2.gaia_source
WHERE parallax < 1 AND 
bp_rp BETWEEN -0.75 AND 2
AND 1 = CONTAINS(
POINT(ra, dec), 
POLYGON({sky_point_list}))
"""

polygon_top10query = polygon_top10query_base.format(columns=columns, sky_point_list=sky_point_list)
print(polygon_top10_query)

polygon_top10query_job = Gaia.launch_job_async(polygon_top10query)
polygon_top10query_job 
print(polygon_top10query_job)

polygon_top10query_results = polygon_top10query_job.get_results()
polygon_top10query_results

polygon_query_base = """
SELECT 
{columns}
FROM gaiadr2.gaia_source
WHERE parallax < 1 AND 
bp_rp BETWEEN -0.75 AND 2
AND 1 = CONTAINS(
POINT(ra, dec), 
POLYGON({sky_point_list}))
"""

polygon_query = polygon_query_base.format(columns=columns, sky_point_list=sky_point_list)
print(polygon_query)

polygon_job = Gaia.launch_job_async(polygon_query)
print(polygon_job)

polygon_results = polygon_job.get_results()
len(polygon_results)

filename = 'gd1_results.fits'
polygon_results.write(filename, overwrite=True)

from os.path import getsize
MB = 1024 * 1024
getsize(filename)/MB

from astropy.table import Table
polygon_results_readin = Table.read(filename)
polygon_results_readin

# last session Peter taking over logging


polygon_results.info()
polygon_results.colnames

type(polygon_results['ra'])
polygon_results.to_pandas().info()


type(polygon_results[0])
polygon_results[0]['ra']
polygon_results['ra'][0]
polygon_results[0][1]
polygon_results[1][0]   _>  not


%matplotlib inline
import matplotlib.pyplot as plt

x = polygon_results['ra']
y = polygon_results['dec']

plt.plot(x, y, 'ko')
plt.xlabel('ra (degree ICRS)')
plt.ylabel('dec (degree ICRS)')
# one of the colab users didn't get a plot here



x = polygon_results['ra']
y = polygon_results['dec']
plt.plot(x, y, 'ko', markersize=0.1, alpha=0.1)
plt.xlabel('ra (degree ICRS)')
plt.ylabel('dec (degree ICRS)')



skycoord = SkyCoord(ra=polygon_results['ra'], dec=polygon_results['dec'])


# could add    frame='icrs'   but that's the default

distance = 8 * u.kpc
radial_velocity= 0 * u.km/u.s

skycoord = SkyCoord(ra=polygon_results['ra'],                       dec=polygon_results['dec'],                    pm_ra_cosdec=polygon_results['pmra'],                    pm_dec=polygon_results['pmdec'],                     distance=distance,                     
radial_velocity=radial_velocity)

transformed = skycoord.transform_to(gd1_frame)

from gala.coordinates import reflex_correct
skycoord_gd1 = reflex_correct(transformed)

x = skycoord_gd1.phi1
y = skycoord_gd1.phi2
plt.plot(x, y, 'ko', markersize=0.1, alpha=0.1)
plt.xlabel('phi1 (degree GD1)')
plt.ylabel('phi2 (degree GD1)')



type(polygon_results)
type(skycoord_gd1)

polygon_results['phi1'] = skycoord_gd1.phi1
polygon_results['phi2'] = skycoord_gd1.phi2
polygon_results.info()

# this showed a bad description for some of us

polygon_results['pm_phi1'] = skycoord_gd1.pm_phi1_cosphi2
polygon_results['pm_phi2'] = skycoord_gd1.pm_phi2
polygon_results.info()


import pandas as pd
results_df = polygon_results.to_pandas()

results_df.shape

results_df.head()


def make_dataframe(table):
   """ transform from ICRS to GD-1
   """
     
  skycoord = SkyCoord(...)
    ...
    return df
    
    # sorry, copy & paste got reformatted, please consult class pages

results_df = make_dataframe(polygon_results)


from astropy.table import Table
results_table = Table.from_pandas(results_df)
type(results_table)


filename = 'gd1_data.hdf'
results_df.to_hdf(filename, 'results_df', mode='w')


# btw, we used gala versioon 1.8.1
gala.__version__


results_df.describe()

# note that parallax smetimes is negative or too positive


x = results_df['pm_phi1']
y = results_df['pm_phi2']
plt.plot(x, y, 'ko', markersize=0.1, alpha=0.1)    

plt.xlabel('Proper motion phi1 (mas/yr GD1 frame)')
plt.ylabel('Proper motion phi2 (mas/yr GD1 frame)')

plt.xlim(-12, 8)
plt.ylim(-10, 10)


phi2 = results_df['phi2']
type(phi2)

phi2_min = -1.0 * u.degree
phi2_max = 1.0 * u.degree
mask = (phi2 > phi2_min)
type(mask)

mask = (phi2 > phi2_min) & (phi2 < phi2_max)
# parens are important

mask.sum()


centerline_df = results_df[mask]

type(centerline_df)


len(centerline_df) / len(results_df)


# now plot for centerline_df instead of results_df and the contrast will be much better


pm1_min = -8.9pm1_max = -6.9pm2_min = -2.2pm2_max =  1.0

pm1_rect, pm2_rect = make_rectangle(pm1_min, pm1_max, pm2_min, pm2_max)


def plot_proper_motion(df):
   x = df['pm_phi1']
       y = df['pm_phi2']
       ...
       
       (again, see the web material)
       


def between(series, low, high):
   """Check whether values are between `low` and `high`."""
    return (series > low) & (series < high)
    



pm1 = results_df['pm_phi1']
pm2 = results_df['pm_phi2']
pm_mask = (between(pm1, pm1_min, pm1_max) & between(pm2, pm2_min, pm2_max))


pm_mask.sum()

selected_df = results_df[pm_mask]
len(selected_df)



x = selected_df['phi1']
y = selected_df['phi2']

plt.plot(x, y, 'ko', markersize=1, alpha=1)
plt.xlabel('phi1 [deg]')

plt.ylabel('phi2 [deg]')

plt.title('Proper motion selection', fontsize='medium')

plt.axis('equal')


def plot_pm_selection(df):
    
    """Plot in GD-1 spatial coordinates the location of the stars
selected by proper motion
"""
# again, see web version; 



plot_pm_selection(selected_df)

filename = 'gd1_data.hdf'
selected_df.to_hdf(filename, 'selected_df')
   # note , no mode='w'  since we want to append
   
   
with pd.HDFStore(filename) as hdf:
      print(hdf.keys())

final homework:      add enterline_df to gd1_data.hdf file




  ----------------------------------------------------------------------------


 
 For anyone using the Colab notebooks, this post explains how to upload local files: https://stackoverflow.com/questions/47320052/load-local-data-files-to-colaboratory
 Let me add to this the full example workflow in colab:
 1) go to colab.reseaarch.google.com
 2) Open Notebook -> Upload     This is where you navigate to your drive and upload our test_setup.ipynb notebook
 3) add a new cell near the top and execute       "!pip install astroquery gala"
 4) add antoher cell and add this file upload code:
      from google.colab import files
     uploaded = files.upload()
     It will allow you to upload files into your colab session.    Upload the gd1_isochrone.hdf5 file, and the  episode_functions.py
5) now execute this whole file cell by cell, and they should all work;   after this you should continue in this notobook
6) Follow the lessons and add cells at the end of this notebook, so you keep the pip install environment, and keep the uploaded files.


 