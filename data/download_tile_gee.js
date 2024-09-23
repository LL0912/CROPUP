var roi=area4
// var s2L2A_cloud=ee.ImageCollection("COPERNICUS/S2_SR").filterDate("2022-10-01","2023-07-01")
// .filterBounds(roi.geometry())
// .filter(ee.Filter.lt('NODATA_PIXEL_PERCENTAGE',25))
// .select('B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12')//b1,B9,B10have little bussiness with crop classification
var year=ee.Number(2019)
var startDay=ee.Date.fromYMD(year,1,1)
var endDay = ee.Date.fromYMD(year,12,31)

var s2L2A_clear=ee.ImageCollection("COPERNICUS/S2_SR").filterDate(startDay,endDay)
.filterBounds(roi)
.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5))
.filter(ee.Filter.lt('NODATA_PIXEL_PERCENTAGE',25))
.select('B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12','QA60')
.map(maskS2clouds)
.select('B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12')

var cdl_startDay=ee.Date.fromYMD(2019,1,1)
var cdl_endDay = ee.Date.fromYMD(2019,12,31)
var CDL = ee.ImageCollection('USDA/NASS/CDL')
                  .filter(ee.Filter.date(cdl_startDay, cdl_endDay))
                  .first()

var cdl_layer = ee.ImageCollection('USDA/NASS/CDL')
                  .filter(ee.Filter.date(cdl_startDay, cdl_endDay))
                  .select('cropland')
                  
var cdl_conf_layer = ee.ImageCollection('USDA/NASS/CDL')
                  .filter(ee.Filter.date(cdl_startDay, cdl_endDay))
                  .select('confidence')
                  
print("cdl:",cdl_layer)

//select the image to process
var s2_img=s2L2A_clear
print(s2_img)

var visualization = {
  min: 0.0,
  max: 1.0,
  bands: ['B4', 'B3', 'B2'],
};
// Map.centerObject(roi, 10)
Map.addLayer(s2L2A_clear.mosaic(), visualization, 'RGB');

function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  
  return image.updateMask(mask).divide(10000).copyProperties(image,["MGRS_TILE","DATATAKE_IDENTIFIER"]);
}

//concatenate by tile without doy selection

//1.get the tile name of the ImageCollection
var tilename=s2_img.aggregate_array('MGRS_TILE').distinct()
var datetime=s2_img.aggregate_array('DATATAKE_IDENTIFIER')
var tiles_list=tilename.getInfo()
print("tilename:",tilename)

//2. filter by tile name
function tile_filter_image(tilelist,imgcol){
var tile_image_list=tilelist.map(function(tile_name){
    // return imgcol.filter(ee.Filter.stringContains('MGRS_TILE',ee.String(tile_name))).toBands().set("MGRS_TILE",tile_name)
    return ee.ImageCollection(imgcol.filterMetadata('MGRS_TILE', 'equals', tile_name)).set("MGRS_TILE",tile_name)
  })
  return tile_image_list
}

//function: get the DATATAKE_IDENTIFIER of the tile
function tile_filter_DATA(tilelist,imgcol){
var tile_image_list=tilelist.map(function(tile_name){
    var imgcolsub= ee.ImageCollection(imgcol.filter(ee.Filter.stringContains('system:index',ee.String(tile_name))).copyProperties(imgcol,["MGRS_TILE","DATATAKE_IDENTIFIER"]));
    var datetime=imgcolsub.aggregate_array('DATATAKE_IDENTIFIER')
    var doy=datetime.map(function(day){
      var year=ee.String(day).slice(5,9)
      var month= ee.Algorithms.If(ee.String(day).slice(9,10).equals('0'), ee.String(day).slice(10,11), ee.String(day).slice(9,11))
      var dday=ee.String(day).slice(11,13)
    var date=ee.Date(year.cat("-").cat(month).cat("-").cat(dday)).getRelative("day","year")
    return date
    })
    return doy
  })
  return tile_image_list
}

var s2_list_1=tile_filter_image(tiles_list,s2_img)
print("s2_list_1",s2_list_1)
var s2_list_time=tile_filter_DATA(tiles_list,s2_img)
print("tile_filter_DATA",s2_list_time)

function exportImage(image, fileName) {
  Export.image.toDrive({
    image: image,
    description: fileName,
    region: image.geometry(),
    fileDimensions:512,
    scale:10,
    crs: "EPSG:4326",
    maxPixels: 1e13,
  });
}

var filter_image=s2_list_1.map(function(sub_image){
  // var tile_name=tilename.get(index)
  var tile_name=sub_image.getString("MGRS_TILE").cat("_image_").cat(ee.String(year)).getInfo()
  // var tile_name=ee.List(tiles_list).get(index)
  // print(tile_name)
  var image=sub_image.toBands()
  exportImage(image, tile_name)
  // print(images)
  
  var CDL_layer = CDL.select('cropland').clip(image.geometry());
  // var CDL_layer = cdl_layer.toBands().clip(image.geometry());
  var tile_name_CDL=sub_image.getString("MGRS_TILE").cat("_CDL_").cat(ee.String(year)).getInfo()
  exportImage(CDL_layer, tile_name_CDL)
  Map.addLayer(CDL_layer, {}, tile_name_CDL);
  
  var CDL_confidence = CDL.select('confidence').clip(image.geometry());
  // var CDL_confidence = cdl_conf_layer.toBands().clip(image.geometry());
  var tile_name_conf=sub_image.getString("MGRS_TILE").cat("_confidence_").cat(ee.String(year)).getInfo()
  exportImage(CDL_confidence, tile_name_conf)
  return image
})
