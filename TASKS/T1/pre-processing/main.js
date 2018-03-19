/*
* PRE-PROCESING THE SHOAIB DATASET
*/

/**
 * SEPARATE SIGNALS ACCORDING TO THEIR SOURCE OF COLLECTION
 * from:    participant_1.csv, participant_2.csv, participant_3.csv, ...
 * to:      wrist_participant_1.csv, left_pocket_participant_1.csv, upper_arm_participant_1.csv, belt_participant_1.csv
 **/

//
var csv = require('csv');
var fs = require('fs');
var store = require('json-fs-store')(__dirname);

/*
return load_object("shoaib", function(object){
    console.log("data.length:", object.length);
});*/

var csvparser = csv.parse({
    delimiter: ','
}, function (err, data) {
    console.log("err:", err);
    console.log("data.length:", data.length);
    store_object("shoaib", data);
});

function store_object(id, object) {
    object.id = id;
    store.add(object, function (err) {
        // called when the file has been written to the
        // /path/to/storage/location/12345.json
        if (err) 
            throw err; // err if the save failed
        }
    );
}

function load_object(id, callback) {
    return store.load(id, function (err, object) {
        if (err) {
            return callback();
        } 
        // err if JSON parsing failed
        // do something with object here
        return callback(object);
    });
}

fs
    .createReadStream('../../../datasets/raw-files/shoaib/participant_1.csv')
    .pipe(csvparser);
