import edu.ucr.cs.bdlab.beast._
import edu.ucr.cs.bdlab.beast.geolite.{EnvelopeNDLite, IFeature}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.LongAccumulator
import org.locationtech.jts.geom.{Coordinate, GeometryFactory, Polygon}
import java.nio.file.Paths
import java.io.{File, FileWriter, FileNotFoundException}
import scala.io.Source
import edu.ucr.cs.bdlab.beast.indexing.RSGrovePartitioner

def calculateGroundTruthValues(sc: SparkContext, range: EnvelopeNDLite, filename: String, factor: Int): (Double, Long, Long) = {
  val geometryFactory = new GeometryFactory()
  val envelopePolygon: Polygon = {
    val coordinates = Array(
      new Coordinate(range.getMinCoord(0), range.getMinCoord(1)),
      new Coordinate(range.getMinCoord(0), range.getMaxCoord(1)),
      new Coordinate(range.getMaxCoord(0), range.getMaxCoord(1)),
      new Coordinate(range.getMaxCoord(0), range.getMinCoord(1)),
      new Coordinate(range.getMinCoord(0), range.getMinCoord(1))
    )
    geometryFactory.createPolygon(coordinates)
  }
  val fileN = filename + ".csv"
  val polygons = sc.spatialFile(s"datasets/$fileN", "point(0,1)", "separator" -> ",")
  val spatialIndexFilename = filename + "_spatial_index"
  val loadedPartitioned = sc.spatialFile(s"index/$spatialIndexFilename", "point(0,1)", "separator" -> ",")
  
  val mbrCount: LongAccumulator = sc.longAccumulator("mbrCount")
  val startTime = System.nanoTime()
  val matchedPolygons = loadedPartitioned.rangeQuery(envelopePolygon, mbrCount)
  val endTime = System.nanoTime()
  val elapsedTime = (endTime - startTime) / 1e6.toLong
  
  val count: Double = matchedPolygons.count()
  val totalNumGeometries = polygons.count() - factor
  val normalizedCount = count / totalNumGeometries
  val mbrCountValue = mbrCount.value

  (normalizedCount, mbrCountValue, elapsedTime)
}

def saveGroundTruthToFile(datasetName: String, queryNumber: String, queryArea: Double, minX: Double, minY: Double, maxX: Double, maxY: Double, areaInt: Double, groundTruth: Double, filename: String, mbrCount: Long, time: Long): Unit = {
  val writer = new FileWriter(new File(filename), true) // Appending mode
  try {
    val groundTruthLine = s"$datasetName;$queryNumber;$queryArea;$minX;$minY;$maxX;$maxY;$areaInt;$groundTruth;$time;$mbrCount"
    writer.write(groundTruthLine + "\n") // Writing to file
  } finally {
    writer.close()
  }
}

val inputFilePath = "rq_newDatasets.csv"
val outputFilePath = "rq_result_p2.csv"
val mbrFilePath = "dataset-summaries.csv"
val startDatasetName = "dataset-0287" // The dataset name to start processing from

val conf = new SparkConf().setAppName("Beast Example").setMaster("local[*]")
val sc = new SparkContext(conf)

// Write the header to the output file
val headerWriter = new FileWriter(new File(outputFilePath), false) // Overwrite mode
try {
  headerWriter.write("datasetName;numQuery;queryArea;minX;minY;maxX;maxY;areaint;cardinality;executionTime;mbrTests\n")
} finally {
  headerWriter.close()
}

// Function to read MBR values from the file
def readMBRValues(datasetName: String, mbrFilePath: String): (Double, Double, Double, Double) = {
  val bufferedSource = Source.fromFile(mbrFilePath)
  for (line <- bufferedSource.getLines()) {
    val cols = line.split(";").map(_.trim)
    if (cols(0) == datasetName) {
      val minX = cols(2).toDouble
      val minY = cols(3).toDouble
      val maxX = cols(4).toDouble
      val maxY = cols(5).toDouble
      bufferedSource.close()
      return (minX, minY, maxX, maxY)
    }
  }
  bufferedSource.close()
  throw new Exception(s"Dataset $datasetName not found in $mbrFilePath")
}

// Read the input file line by line
val bufferedSource = Source.fromFile(inputFilePath)
var startProcessing = false

for ((line, lineNumber) <- bufferedSource.getLines().drop(1).zipWithIndex) { // Skip the header line
  val cols = line.split(",").map(_.trim)
  if (cols.length >= 7) {
    try {
      val datasetName = cols(0).stripMargin.replaceAll("\"", "")
      if (datasetName == startDatasetName) {
        startProcessing = true
      }
      if (startProcessing) {
        val minX = cols(3).replace("\"", "").toDouble
        val minY = cols(4).replace("\"", "").toDouble
        val maxX = cols(5).replace("\"", "").toDouble
        val maxY = cols(6).replace("\"", "").toDouble
        val queryNumber = cols(1).stripMargin.replaceAll("\"", "")
        
        println(s"Dataset Name: $datasetName")
        println(s"QNumber: $queryNumber")
        println()

        val range = new EnvelopeNDLite(2, minX, minY, maxX, maxY)
        val queryArea = (maxX - minX) * (maxY - minY)
        
        try {
          val (datasetMinX, datasetMinY, datasetMaxX, datasetMaxY) = readMBRValues(datasetName, mbrFilePath)
          
          val a = if (minX > datasetMinX) minX else datasetMinX
          val b = if (minY > datasetMinY) minY else datasetMinY
          val c = if (maxX < datasetMaxX) maxX else datasetMaxX
          val d = if (maxY < datasetMaxY) maxY else datasetMaxY

          var areaInt = (c - a) * (d - b)
          if (areaInt < 0) {
            areaInt = 0
          }

          val (groundTruthValues, mbrCount, time) = calculateGroundTruthValues(sc, range, datasetName, 0)
          saveGroundTruthToFile(datasetName, queryNumber, queryArea, minX, minY, maxX, maxY, areaInt, groundTruthValues, outputFilePath, mbrCount, time)
        } catch {
          case e: FileNotFoundException =>
            println(s"File $datasetName not found. Skipping to the next dataset.")
        }
      }
    } catch {
      case e: NumberFormatException =>
        println(s"Error parsing line $lineNumber: ${e.getMessage}")
    }
  } else {
    println(s"Error parsing line $lineNumber: Insufficient columns")
  }
}
bufferedSource.close()
sc.stop()

