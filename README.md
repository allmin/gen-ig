# gen-ig
GEN-IG : a GENeric toolkit for schema-driven Insight Generation
- In this repository, we provide a fully customizable insight generator that can be used for any insight generation application.
- installation: Python3.8 -> pip install -r requirements.txt will install all the dependencies
- The main scripts are: 
-- 1) generate_insight_lib.py: To generate an insight library. It needs a system_definition file
-- 2) score_insight_lib.py: to score the insight library. It requires data, system definitions and an insight_library file
-- 3) how_to_say.py: cleans the text suitable for presentation
-- 4) recommend_insights: recommends N=20 diverse insights
- The system definitions help to define the entire IG system. In this repository, system definitions are by default provided for an abstract "example1" use case that contains measurements and contexts. 
- Note: 
-- The end-to-end.py runs all the pipeline successively.
-- The config.py comes handy in defining important configurations of the insight generator

## config.py
- Mention the system name in the config.py file as "example", or anything else to choose the appropriate system
- Name your system difinition files, data files, etc with the same suffix (see example systems in ./systems)
- Mention the data_file_format (The format in which the data is presented to the gen-ig. preferably pickle)
- scope_name applies when each IG is run to a specific customer/ user/ hospital. 
-- if scope_name is defined provide the data for each scope in .systems/<system_name>/data_<scope_name>_<system_name>.pickle
-- if scope_name is not defined given provide the data for each scope in .systems/<system_name>/data_<system_name>.pickle
- The unknown_phrase is assigned a phrase that is generated when comparisons are not possible due to missing/inadequate data
-- assign it to None to fill it with the default text <<unknown>>

## system_definition
- Each system has a subfolder inside the ./systems folder eg: ./systems/igt
- An excel file eg: systems/igt/system_definition_igt.xlsx defines all the necessary information to generate and score the insights
- Each sheet in this excel file defines each aspect of the IG
- Let us see the individual sheets.

### sheet1: schemas
- This sheet defines the structure of schemas

|SNo|Component|Description|example|
|---|---------|-----------|-------|
|1|schema num|Specify the identifier for the schema|1|
|2|template|Specify the structure for the schema. It can have zero or more common contexts, two comparative contexts. the second comparative context could be same as the first, an inversion of the first or a benchmark | {{period:1}} the {{measurement}} was {{comparison}} {{period:2}} <br><br> {{period:1}} the {{measurement}} is {{comparison}} than {{!period:2}} <br><br> {{period:1}} the {{measurement}} is {{comparison}} than {{measurement_benchmark:2}} <br><br> optionally, you may include {{mean:1}}, {{mean:2}} and {{percentage}} to display mean of first context, mean of second context and percentage difference of the first and second contexts respectively.|
|3|silent_contexts|input contexts that needs to be considered in the insight, but not present in the text|['user_id','period:2']
|4|applicable_items| it is a dictionary containing the measurements applicable to the schema. use 'all' to use all measurements| {'measurement':'all'} / {'measurement':['room utilization', 'on-time starts', 'turn-around times']}
|5|example| representative examples for each schema| This week the room utilization is less than last week <br><br> On Mondays the room utilization is less than other days <br><br> On Mondays the room utilization is greater than 85%|
|6|scoring type| the type of scoring mechanisms for each schema| 'ks' for kolmogorov-smirnov, 'mw' for man whitney and  'benchmark' to compare the first context with a benchmark| 
|7|tag| an optional string to be tagged to to each insight of the schema. can be useful for filtering insights| 'long_term', 'priority1'| 

- Note: to denote the comparative contexts, the first context has a suffix of :1 and the second context gets a suffix :2

### sheet2: contexts
- In this sheet we mention all the contexts being compared in the insight tool.
- Note: Avoid using the following reserved names for contexts:
-- be
-- measurement
-- comparison

|SNo|Component|Description|example|
|---|---------|-----------|-------|
|1|context|Specify the name of each context mentioned in the schema. No need to specify the "measurement_benchmark" context and inverted contexts (!period)|period|

### sheet3: measurements
- The measurements, their phrases and benchmarks are defined in this sheet

|SNo|Component|Description|example|
|---|---------|-----------|-------|
|1|measurement		    |Here we give a user defined name for a measurement|calories-hourly|
|2|column_in_data		    |Here we mention the column in the data that holds the measure|calories|
|3|phrase	            |The phrase that describes the measure in the insight text|calories burnt in an hour|
|4|regular comparative phrase   |a phrase to describe each comparison states|['less than', 'within the range:', 'greater than']|
|5|benchmark	        |the different categories that a comparison could fall with respect to a benchmark described as an executable python list. The keywords 'measurement' denotes any measure that is being benchmarked|[measurement<18,(measurement >= 18 & measurement <= 20),measurement>20]|
|6|benchmark comparative phrase   |a phrase to describe each of the above benchmark states|['less than', 'within the range:', 'greater than']|	
|7|benchmark text       |This is the text that describes the benchmark value|['100 calories','100 - 200 calories','200 calories']|
|8|group_by|defines on what coulmn(s) the measurement needs to be aggregated|day_date / ['day_date','day_time']. Prefix a '!' for groupings that are not imestamp related|
|9|aggregate|specify the aggregation method|sum|
|10|tolerance|specify how much minimum difference is considered significant|10|
|11|unit|specify the unit|s|

### sheet4: exclusions
- The exclusions allow us to eliminate certain insight occurrences, for example, we don’t want to compare room utilization in January vs any other context.
- A initial rough run on the system definition gives rise to a library file. We may skim through this library file to define the exclusions for subsequent runs

|SNo|Component|Description|example|
|---|---------|-----------|-------|
|1|schema num|The schema number from which statements should be excluded. It can be mentioned as 'all' without quotes to apply to all schemas.
|2|target column|A column in the library file in which the exclusion criteria must be applied|insight_text|
|3|match|a regex pattern to match the entries of the above column to be excluded|.* January .* utilization of the room .* #this excludes all statements that compare January room utilizations with any other contexts|
		
### sheet5: c_<context_name> Eg: c_period
- For each context mentioned in the context sheet, we need to create a c_<context> sheet to define the contexts as follows
- in this sheet, we describe the subcontexts, entities, phrases, comparability, etc
- for a very complex context see system_definition_igt.xlsx/c_period
- for a simple context see system_definition_igt.xlsx/c_population
- specify the tense as [SP, PC, PP, PPC, SPa, PaC, PaP, PaPC, Future] for  [Simple Present + present Continuous + Present Perfect + Present Perfect Continuous + Simple Past + Past Continuous + Past Perfect + Past Perfect Continuous + Future]

|SNo|Component|Description|example|
|---|---------|-----------|-------|
|1|name|an identifier given to a subcontext|room1|
|2|entity|The members of the context|['low',  'high']|
|3|sql|a pandas query to filter the contexts from the data|["data['categorical room utilization'] = {}".format(i) for i in entities]|
|4|prepositional phrase|A prepositional phrase for each entity in the entity column|['when the room was utilized less', 'when the room was utilized optimally']|
|5|inverse query|a pandas query to filter the contexts from the inverse contexts of corresponding contexts in 'sql' column|["data['categorical room utilization'] != {}".format(i) for i in entities]|
|6|inversion phrase|This is the phrase to denote inversions to any given entity|when not|
|7|pair id|comparable subcontests have same pair id. it is different otherwise|1|
|8|consecutive constrain|if true, will compare only consecutive entities. For example January vs february and not january vs march|False|


## preprocessing script.
optionally, a file names preprocess_data.py can be created in the correspondin system folder to perform preprocessing.


