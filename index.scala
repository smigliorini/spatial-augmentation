import edu.ucr.cs.bdlab.beast._
import edu.ucr.cs.bdlab.beast.geolite.{EnvelopeNDLite, IFeature}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.apache.spark.util.LongAccumulator
import org.locationtech.jts.geom.{Coordinate, GeometryFactory, Polygon}
import java.io.{File, FileWriter, FileNotFoundException}
import scala.io.Source
import edu.ucr.cs.bdlab.beast.indexing.RSGrovePartitioner


// Create SparkConf
val conf = new SparkConf().setAppName("Beast Example").setMaster("local[*]")



// Create SparkContext
val sc = new SparkContext(conf)



// Folder containing dataset file (singular now)
val datasetFolder = "datasets"



// Output folder for spatial indices
val outputFolder = "index"



// Create output folder if it doesn't exist
new File(outputFolder).mkdir()



// Specify the dataset file (replace "dataset-0001.csv" with your actual file)
val datasetName = "dataset-0537.csv" // Replace with your dataset filename



// Load dataset
val polygons = sc.spatialFile(s"$datasetFolder/$datasetName", "point(0,1)", "separator" -> ",")

  

// Spatial partitioning
val partitionedFeatures: RDD[IFeature] = polygons.spatialPartition(classOf[RSGrovePartitioner])

 
// Save spatial index
partitionedFeatures.writeSpatialFile(s"$outputFolder/${datasetName.stripSuffix(".csv")}_spatial_index", "point(0,1)", "separator" -> ",")

// Stop SparkContext
sc.stop()
