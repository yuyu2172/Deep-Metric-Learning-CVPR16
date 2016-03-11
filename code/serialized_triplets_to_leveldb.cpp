#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <lmdb.h>

#include <sys/stat.h>
#include <stdint.h>

//#include <caffe/proto/caffe.pb.h>
#include "caffe.pb.h"

#include "mex.h"

class DB {
public:
    DB(const char* db_path, bool use_leveldb_) : use_leveldb(use_leveldb_), count(0), batch(NULL) {
        // Open db
        if (use_leveldb) {  // leveldb
            LOG(INFO) << "Opening leveldb " << db_path;
            leveldb::Options options;
            options.error_if_exists = true;
            options.create_if_missing = true;
            options.write_buffer_size = 268435456;
            leveldb::Status status = leveldb::DB::Open(options, db_path, &db);
            CHECK(status.ok()) << "Failed to open leveldb " << db_path
                    << ". Is it already existing?";
            batch = new leveldb::WriteBatch();
        } else {  // lmdb
            LOG(INFO) << "Opening lmdb " << db_path;
            CHECK_EQ(mkdir(db_path, 0744), 0)
            << "mkdir " << db_path << "failed";
            CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
            CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
            << "mdb_env_set_mapsize failed";
            CHECK_EQ(mdb_env_open(mdb_env, db_path, 0, 0664), MDB_SUCCESS)
            << "mdb_env_open failed";
            CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
            CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
            << "mdb_open failed. Does the lmdb already exist? ";
        }
    }
    
    
    ~DB() {
        // write the last batch
        if (count % 1000 != 0) {
            if (use_leveldb) {
                db->Write(leveldb::WriteOptions(), batch);
            } else {  // lmdb
                CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
            }
            LOG(ERROR) << "Processed " << count << " files.";
        }

        // close db
        if (use_leveldb) {
            delete batch;
            delete db;
        } else {  // lmdb
            mdb_close(mdb_env, mdb_dbi);
            mdb_env_close(mdb_env);
        }
        LOG(INFO) << "Database closed.";
    }
    
    void write(const std::string& keystr, const std::string& value) {
        if (use_leveldb) {  // leveldb
            batch->Put(keystr, value);
        } else {  // lmdb
            mdb_data.mv_size = value.size();
            mdb_data.mv_data = const_cast<char*>(&value[0]);
            mdb_key.mv_size = keystr.size();
            mdb_key.mv_data = const_cast<char*>(&keystr[0]);
            CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
            << "mdb_put failed";
        }

        if (++count % 1000 == 0) {
            // Commit txn
            if (use_leveldb) {  // leveldb
                db->Write(leveldb::WriteOptions(), batch);
                delete batch;
                batch = new leveldb::WriteBatch();
            } else {  // lmdb
                CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
                << "mdb_txn_commit failed";
                CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
                << "mdb_txn_begin failed";
            }
            LOG(ERROR) << "Processed " << count << " files.";
        }
    }

private:
    const bool use_leveldb;

    int count;

    // lmdb
    MDB_env *mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn;

    // leveldb
    leveldb::DB* db;
    leveldb::WriteBatch* batch;
};


void process(uint8_t* images, int num_images, int img_width, int img_height,
    int32_t* serialized_triplets, int num_examples, const char* db_filename) {

    // Temporary storage for assembling image triplet
    const size_t img_size = 3 * img_width * img_height;
    uint8_t* serialized_triplet_buf = new uint8_t[img_size];

    const int kMaxKeyLength = 10;
    char key[kMaxKeyLength];
    std::string value;

    caffe::Datum datum;
    datum.set_channels(3);
    datum.set_height(img_height);
    datum.set_width(img_width);

    DB db(db_filename, false);

    for (int i = 0; i < num_examples; ++i) {
        size_t id = serialized_triplets[i];
        if (id < 0 || id >= num_images){
            std::cout << "Image id out of bounds: id: " << id << "num_images: " << num_images << std::endl;
            goto out;
        }
        uint8_t* img = &images[id * img_size];

        memcpy(serialized_triplet_buf, img, img_size);

        datum.set_data(serialized_triplet_buf, img_size);
        datum.SerializeToString(&value);

        snprintf(key, kMaxKeyLength, "%08d", i);
        
        db.write(std::string(key), value);
    }

    out:
    delete [] serialized_triplet_buf;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
    if (nrhs != 3) {
        mexErrMsgTxt("Must have 3 inputs"); 
    }
    if (nlhs != 0) {
        mexErrMsgTxt("Must have 0 outputs");
    }

    // Read images
    if (!mxIsUint8(prhs[0]) || mxGetNumberOfDimensions(prhs[0]) != 4) {
        mexErrMsgTxt("images must be a 4D uint8 array");
    }
    uint8_t* images = (uint8_t*) mxGetData(prhs[0]);
    const mwSize* images_size = mxGetDimensions(prhs[0]);
    int width = images_size[0];
    int height = images_size[1];
    int num_channels = images_size[2];
    if (num_channels != 3) {
        mexErrMsgTxt("Expected 3 channels for images");
    }
    int num_images = images_size[3];
    int img_length = width * height * num_channels;
    std::cout << "Image info: width: " << width << " height: " << height << " channels: " << num_channels << " num: " << num_images << std::endl;

    // Read serialized triplets
    if (!mxIsInt32(prhs[1]) || mxGetN(prhs[1]) != 1) {
        mexErrMsgTxt("triplets must have type int32 and have 1 column");
    }
    int32_t* serialized_triplets = (int32_t*) mxGetData(prhs[1]);
    int num_examples = mxGetM(prhs[1]);

    // Read db filename
    char* db_filename = mxArrayToString(prhs[2]);
    printf("%s\n", db_filename);

    google::InitGoogleLogging("triplets_to_caffedb");
    process(images, num_images, width, height, serialized_triplets, num_examples, db_filename);
    
    mxFree(db_filename);
}
