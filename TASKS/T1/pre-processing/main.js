/*
* PRE-PROCESING THE SHOAIB DATASET
*/

/**
 * SEPARATE SIGNALS ACCORDING TO THEIR SOURCE OF COLLECTION
 * from:    participant_1.csv, participant_2.csv, participant_3.csv, ...
 * to:      wrist_participant_1.csv, left_pocket_participant_1.csv, upper_arm_participant_1.csv, belt_participant_1.csv
 **/

var args = process
        .argv
        .slice(2),
    FILENAME = null;

if (args.length >= 1) {
    for (var i = 0; i < args.length; i++) {
        if (args[i].indexOf("file") != -1) {
            var array = args[i].split("=");
            if (array.length == 2) {
                FILENAME = array[1];
            }
        }
    }
} else {
    throw Error("Invalid Arguments! Ex: node main.js file=<filename>");
}

//
var csv = require('csv'),
    fs = require('fs'),
    store = require('json-fs-store')(__dirname),
    dataset = null,
    csvparser = csv.parse({
        delimiter: ','
    }, function (err, data) {
        console.log("err:", err);
        console.log("data.length:", data.length);
        store_object(FILENAME, data);
        dataset = data;
    });

const INDEX_LABEL = 69;
const NR_OF_COLUMNS = 13;

preprocessdataset();

//
function preprocessdataset() {
    return load_object(FILENAME, function (object) {
        if (!object) {
            console.log("reading-file");
            var stream = fs
                .createReadStream("../../../datasets/raw-files/shoaib/" + FILENAME + ".csv")
                .pipe(csvparser);

            return stream.on('finish', function () {
                console.log("golo!");
                //proceed(dataset);
                setTimeout(function () {
                    return preprocessdataset(dataset);
                }, 2000);
            });
        } else {
            console.log("in-memory");
        }

        /*
        // count nr of columns per sensor-placement
            // 14 columns per each sensor-placement
            // 70 columns in total
            // ax, ay, az,lx,ly,lz,gx,gy,gz,mx,my,mz,label*/

        var header = "ax, ay, az,lx,ly,lz,gx,gy,gz,mx,my,mz,label";
        var files = ["left_pocket.csv", "right_pocket.csv", "wrist.csv", "upper_arm.csv", "belt.csv"];
        var participant_data = [new String(header), new String(header), new String(header), new String(header), new String(header)];
        // label is at 69th position
        for (var i = 3; i < object.length; i++) {

            var placementData = String(object[i]).split(',,');
            const label = object[i][INDEX_LABEL];
            for (var j = 0; j < placementData.length; j++) {
                var data = String(placementData[j]).split(',');
                if (j < placementData.length - 1) 
                    data.push(label);
                participant_data[j] += "\n" + String(data.slice(1));
            }
        }

        console.log(participant_data[1]);
        // write files
        var index = 0;
        var writeFiles = function () {
            return fs.writeFile("valid-dataset/" + FILENAME + "_" + files[index], participant_data[index], function (err) {
                if (err) {
                    return console.log(err);
                }
                console.log("The file", files[index], " was saved!");
                if (index < 5) {
                    index++;
                    writeFiles();
                } else {
                    console.log("finished");
                }
            });
        }
        writeFiles();

    });
}

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
        // err if JSON parsing failed do something with object here
        return callback(object);
    });
}
