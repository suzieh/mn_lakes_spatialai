--This file contains SQL queries for creating buffer feature vectors
--  for the MN OSM data and MN DNR assessed lakes

-- NOTES TO RUN THIS SCRIPT (requires a machine set up with PostGIS)
--     $ conda activate spatial
--     $ pg_ctl -D mylocal_db -l logfile start
--     $ psql inner_final username
--         note: to see all tables in DB: \dt+
--         note: quit out of psql: \q
--     $ pg_ctl -D mylocal_db stop


-- Create postgis extension
CREATE EXTENSION postgis;

-- Create table for lake buffers (note: previously created in python)
DROP TABLE IF EXISTS lake_buffers;
CREATE TABLE lake_buffers(
    u_id INTEGER PRIMARY KEY,
    lake_id VARCHAR(10) NOT NULL,
    lake_name VARCHAR(40) NOT NULL,
    buffer_size INTEGER NOT NULL,
    geometry geometry(Polygon,26915) NOT NULL);
---- note that geometry is originally in NAD83 format, which uses degree coordinates

-- Read in buffer data from CSV - should return 'COPY 1840'
COPY lake_buffers(u_id, lake_id, lake_name, buffer_size, geometry)
    FROM '/Users/susanhoops/Desktop/CSCI8980/Final_Project/data/assessed_lakes_buffers.csv'
    DELIMITER ','
    CSV HEADER;


-- Create index for buffers, transform geometries to compatible format (WGS84)
SELECT AddGeometryColumn('lake_buffers','buffer_4326', 4326, 'POLYGON', 2);
UPDATE lake_buffers SET buffer_4326 = ST_Transform(geometry, 4326);
CREATE INDEX "lake_buffers_buffer_idx" ON lake_buffers USING gist(buffer_4326);
---- now geometry is same SRID as OSM features tables. Can check this with:
----   SELECT Find_SRID('public', 'lake_buffers', 'buffer_4326');
----   SELECT Find_SRID('public', 'point_features', 'wkb_geometry');

-- Create indices for OSM feature tables
CREATE INDEX "line_features_wkb_geometry_geom_idx" ON line_features USING gist(wkb_geometry);
CREATE INDEX "point_features_wkb_geometry_geom_idx" ON point_features USING gist(wkb_geometry);
CREATE INDEX "polygon_features_wkb_geometry_geom_idx" ON polygon_features USING gist(wkb_geometry);


-- Create empty table for geo context
DROP TABLE IF EXISTS geographic_features;
CREATE TABLE geographic_features(
    u_id BIGSERIAL PRIMARY KEY,
    lake_id VARCHAR(10) NOT NULL,
    geom_type TEXT NOT NULL,
    geo_feature TEXT NOT NULL,
    feature_type TEXT NOT NULL,
    buffer_size INTEGER NOT NULL,
    value  DOUBLE PRECISION);

-- POINT OSM features to geo context
INSERT INTO geographic_features (u_id, lake_id, geom_type, geo_feature, feature_type, buffer_size, value)
    SELECT nextval('geographic_features_u_id_seq'::regclass),
           lb.lake_id,
           'point' AS geom_type,
           pf.geo_feature,
           pf.feature_type,
           lb.buffer_size,
           ST_NPoints(ST_Transform(ST_Intersection(lb.buffer_4326, pf.wkb_geometry), 3857)) AS value
    FROM lake_buffers lb, point_features pf
    WHERE ST_Intersects(lb.buffer_4326, pf.wkb_geometry);
---- inserts 11931 intersections

-- LINE OSM features to geo context
INSERT INTO geographic_features (u_id, lake_id, geom_type, geo_feature, feature_type, buffer_size, value)
    SELECT nextval('geographic_features_u_id_seq'::regclass),
           lb.lake_id,
           'line' AS geom_type,
           lf.geo_feature,
           lf.feature_type,
           lb.buffer_size,
           ST_Length(ST_Transform(ST_Intersection(lb.buffer_4326, lf.wkb_geometry), 3857)) AS value
    FROM lake_buffers lb, line_features lf
    WHERE ST_Intersects(lb.buffer_4326, lf.wkb_geometry);
---- inserts 272280 intersections

-- POLYGON OSM features to geo context
INSERT INTO geographic_features (u_id, lake_id, geom_type, geo_feature, feature_type, buffer_size, value)
    SELECT nextval('geographic_features_u_id_seq'::regclass),
           lb.lake_id,
           'polygon' AS geom_type,
           plf.geo_feature,
           plf.feature_type,
           lb.buffer_size,
           ST_Area(ST_Transform(ST_Intersection(lb.buffer_4326, plf.wkb_geometry), 3857)) AS value
    FROM lake_buffers lb, polygon_features plf
    WHERE ST_Intersects(lb.buffer_4326, plf.wkb_geometry);
---- inserts 94152 intersections

-- COPY tables out
SELECT COUNT(*) FROM geographic_features;
---- should get 378363 rows
---- copy out lake_buffers to shapefile using ogr2ogr:
----   $ ogr2ogr -f "ESRI Shapefile" mybuffers PG:"dbname=inner_final user=suzie password=XXXXXXXXXXX" lake_buffers
---- copy out geographic_features to CSV:
COPY geographic_features TO '/Users/susanhoops/Desktop/CSCI8980/Final_Project/data/osm_features.csv' csv header;
