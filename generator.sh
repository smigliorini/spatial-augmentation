#!/bin/bash

# Path to your CSV file
csv_file="dataset-summaries2.csv"

LOG_FILE="commands.log"

# Function to execute the generator script
execute_generator() {
    distribution=$1
    cardinality=$2
    polysize=$3
    output_file=$4
    maxseg=$5
    width=$6
    height=$7
    affinematrix=$8

    if [ "$distribution" = "uniform" ]; then # uniform
        command="./generator.py distribution=uniform cardinality=$cardinality dimensions=2 geometry=box polysize=$polysize maxseg=$maxseg format=csv affinematrix=$affinematrix maxsize=$width,$height affinematrix=$affinematrix"
        
        
    elif [ "$distribution" = "diagonal" ]; then # diagonal
        command="./generator.py distribution=diagonal cardinality=$cardinality dimensions=2 percentage=0.5 buffer=0.5 geometry=box polysize=$polysize maxseg=$maxseg format=csv affinematrix=$affinematrix maxsize=$width,$height affinematrix=$affinematrix"
        
        
    elif [ "$distribution" = "gaussian" ]; then # gaussian
        command="./generator.py distribution=gaussian cardinality=$cardinality dimensions=2 geometry=box polysize=$polysize maxseg=$maxseg format=csv affinematrix=$affinematrix maxsize=$width,$height affinematrix=$affinematrix"
        
        
    elif [ "$distribution" = "sierpinski" ]; then # sierpinski
        command="./generator.py distribution=sierpinski cardinality=$cardinality dimensions=2 geometry=box polysize=$polysize maxseg=$maxseg format=csv affinematrix=$affinematrix maxsize=$width,$height affinematrix=$affinematrix"
        
        
    elif [ "$distribution" = "bit" ]; then # bit
        command="./generator.py distribution=bit cardinality=$cardinality dimensions=2 probability=0.2 digits=10 geometry=box polysize=$polysize maxseg=$maxseg format=csv affinematrix=$affinematrix maxsize=$width,$height affinematrix=$affinematrix"
        
        
    elif [ "$distribution" = "parcel" ]; then # parcel
        command="./generator.py distribution=parcel cardinality=$cardinality dimensions=2 srange=0.5 dither=0.5 polysize=$polysize maxseg=$maxseg format=csv affinematrix=$affinematrix affinematrix=$affinematrix"
    else
        echo "Unknown distribution: $distribution"
        return
    fi

    # Execute the command and save to the output file
    $command  > "$output_file"
    # Run the command and save it to the log file
    echo "$command" >> "$LOG_FILE"
}

# Read the CSV file and execute the generator for each row
total_datasets=$(wc -l < "$csv_file")
log_file="commands.log"

echo "Generating datasets..."

# Loop through each row in the CSV file
while IFS=";" read -r datasetName distribution x1 y1 x2 y2 num_features size num_points avg_area avg_side_length_0 avg_side_length_1 E0 E2; do
    echo "Processing dataset: $datasetName"
    echo "Distribution: $distribution"
    
    output_file="$datasetName.csv"
    
    # MA settings
    a1=$(echo "$x2 - $x1" | bc)
    a3=$x1
    a5=$(echo "$y2 - $y1" | bc)
    a6=$y1
    matrix="$a1,0,$a3,0,$a5,$a6"
    
    # Call the execute_generator function with the row data
    execute_generator "$distribution" "$num_features" "$avg_area" "$output_file" "4" "$avg_side_length_0" "$avg_side_length_1" "$matrix"
done < <(tail -n +2 "$csv_file")

echo "Generation complete"
