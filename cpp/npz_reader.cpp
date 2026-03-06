#include "npz_reader.h"
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <zlib.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace cls {

// Parse .npy header to extract dtype and shape
static void parse_npy_header(const char* buf, size_t len,
                              std::string& dtype, std::vector<size_t>& shape, int& elem_size) {
    std::string header(buf, len);

    // Find 'descr' value
    auto pos = header.find("'descr'");
    if (pos == std::string::npos) pos = header.find("\"descr\"");
    if (pos == std::string::npos) throw std::runtime_error("No descr in npy header");
    pos = header.find('\'', pos + 7);
    if (pos == std::string::npos) pos = header.find('"', pos);
    char quote = header[pos];
    size_t end = header.find(quote, pos + 1);
    dtype = header.substr(pos + 1, end - pos - 1);

    // Determine element size from dtype
    if (dtype == "<f8" || dtype == "<d") elem_size = 8;
    else if (dtype == "<f4" || dtype == "<f") elem_size = 4;
    else if (dtype == "<i8" || dtype == "<q") elem_size = 8;
    else if (dtype == "<i4" || dtype == "<i" || dtype == "<l") elem_size = 4;
    else if (dtype == "<i2") elem_size = 2;
    else if (dtype == "<u1" || dtype == "|u1" || dtype == "|b1") elem_size = 1;
    else if (dtype.substr(0, 2) == "|S") elem_size = std::stoi(dtype.substr(2));
    else if (dtype.substr(0, 2) == "<U") elem_size = 4 * std::stoi(dtype.substr(2));
    else if (dtype == "bool" || dtype == "|b1") elem_size = 1;
    else {
        // Try to extract size from last chars
        elem_size = 8; // default
    }

    // Find 'shape' value
    shape.clear();
    pos = header.find("'shape'");
    if (pos == std::string::npos) pos = header.find("\"shape\"");
    if (pos == std::string::npos) throw std::runtime_error("No shape in npy header");
    pos = header.find('(', pos);
    end = header.find(')', pos);
    std::string shape_str = header.substr(pos + 1, end - pos - 1);

    // Parse shape tuple
    if (!shape_str.empty()) {
        std::istringstream ss(shape_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // Trim whitespace
            tok.erase(0, tok.find_first_not_of(" \t"));
            tok.erase(tok.find_last_not_of(" \t") + 1);
            if (!tok.empty()) {
                shape.push_back(std::stoull(tok));
            }
        }
    }
    if (shape.empty()) shape.push_back(1); // scalar
}

static NpyArray parse_npy(const char* data, size_t total_len) {
    NpyArray arr;

    // Check magic
    if (total_len < 10 || (unsigned char)data[0] != 0x93 ||
        data[1] != 'N' || data[2] != 'U' || data[3] != 'M' ||
        data[4] != 'P' || data[5] != 'Y') {
        throw std::runtime_error("Invalid .npy magic");
    }

    uint8_t major = data[6];
    // uint8_t minor = data[7];
    size_t header_len;
    size_t header_offset;

    if (major == 1) {
        header_len = *(uint16_t*)(data + 8);
        header_offset = 10;
    } else {
        header_len = *(uint32_t*)(data + 8);
        header_offset = 12;
    }

    parse_npy_header(data + header_offset, header_len,
                     arr.dtype, arr.shape, arr.elem_size);

    size_t data_offset = header_offset + header_len;
    size_t data_len = total_len - data_offset;

    arr.raw_data.assign(data + data_offset, data + data_offset + data_len);
    return arr;
}

int64_t NpyArray::as_scalar_int() const {
    if (dtype == "<i8" || dtype == "<q") return *reinterpret_cast<const int64_t*>(raw_data.data());
    if (dtype == "<i4" || dtype == "<i" || dtype == "<l") return *reinterpret_cast<const int32_t*>(raw_data.data());
    if (dtype == "<i2") return *reinterpret_cast<const int16_t*>(raw_data.data());
    if (dtype == "<u1" || dtype == "|u1") return *reinterpret_cast<const uint8_t*>(raw_data.data());
    if (dtype == "<f8") return (int64_t)*reinterpret_cast<const double*>(raw_data.data());
    if (dtype == "<f4") return (int64_t)*reinterpret_cast<const float*>(raw_data.data());
    throw std::runtime_error("Cannot convert dtype " + dtype + " to int");
}

std::string NpyArray::as_string() const {
    if (dtype.substr(0, 2) == "|S") {
        // Fixed-width byte string - find null terminator or use full length
        size_t len = raw_data.size();
        for (size_t i = 0; i < raw_data.size(); i++) {
            if (raw_data[i] == '\0') { len = i; break; }
        }
        return std::string(raw_data.data(), len);
    }
    if (dtype.substr(0, 2) == "<U") {
        // UTF-32 LE string
        int n_chars = std::stoi(dtype.substr(2));
        std::string result;
        const uint32_t* chars = reinterpret_cast<const uint32_t*>(raw_data.data());
        for (int i = 0; i < n_chars; i++) {
            if (chars[i] == 0) break;
            if (chars[i] < 128) result += (char)chars[i];
        }
        return result;
    }
    return std::string(raw_data.data(), raw_data.size());
}

std::vector<double> NpyArray::to_float64_vec() const {
    size_t n = numel();
    std::vector<double> result(n);
    if (dtype == "<f8" || dtype == "<d") {
        std::memcpy(result.data(), raw_data.data(), n * sizeof(double));
    } else if (dtype == "<f4" || dtype == "<f") {
        const float* src = reinterpret_cast<const float*>(raw_data.data());
        for (size_t i = 0; i < n; i++) result[i] = src[i];
    } else if (dtype == "<i4" || dtype == "<i" || dtype == "<l") {
        const int32_t* src = reinterpret_cast<const int32_t*>(raw_data.data());
        for (size_t i = 0; i < n; i++) result[i] = src[i];
    } else if (dtype == "<i8" || dtype == "<q") {
        const int64_t* src = reinterpret_cast<const int64_t*>(raw_data.data());
        for (size_t i = 0; i < n; i++) result[i] = src[i];
    }
    return result;
}

std::vector<int32_t> NpyArray::to_int32_vec() const {
    size_t n = numel();
    std::vector<int32_t> result(n);
    if (dtype == "<i4" || dtype == "<i" || dtype == "<l") {
        std::memcpy(result.data(), raw_data.data(), n * sizeof(int32_t));
    } else if (dtype == "<i8" || dtype == "<q") {
        const int64_t* src = reinterpret_cast<const int64_t*>(raw_data.data());
        for (size_t i = 0; i < n; i++) result[i] = (int32_t)src[i];
    } else if (dtype == "<f8" || dtype == "<d") {
        const double* src = reinterpret_cast<const double*>(raw_data.data());
        for (size_t i = 0; i < n; i++) result[i] = (int32_t)src[i];
    } else if (dtype == "<f4" || dtype == "<f") {
        const float* src = reinterpret_cast<const float*>(raw_data.data());
        for (size_t i = 0; i < n; i++) result[i] = (int32_t)src[i];
    }
    return result;
}

// ZIP local file header signature
static const uint32_t ZIP_LOCAL_MAGIC = 0x04034b50;

NpzFile load_npz(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0)
        throw std::runtime_error("Cannot stat: " + path);
    long file_size = st.st_size;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Cannot open: " + path);

    std::vector<char> file_data(file_size);
    long total_read = 0;
    while (total_read < file_size) {
        ssize_t n = read(fd, file_data.data() + total_read, file_size - total_read);
        if (n <= 0) { close(fd); throw std::runtime_error("Failed to read: " + path); }
        total_read += n;
    }
    close(fd);

    NpzFile npz;
    size_t offset = 0;

    while (offset + 30 <= (size_t)file_size) {
        uint32_t sig;
        std::memcpy(&sig, file_data.data() + offset, 4);
        if (sig != ZIP_LOCAL_MAGIC) break;

        uint16_t compression, fname_len, extra_len;
        uint32_t comp_size32, uncomp_size32;

        std::memcpy(&compression, file_data.data() + offset + 8, 2);
        std::memcpy(&comp_size32, file_data.data() + offset + 18, 4);
        std::memcpy(&uncomp_size32, file_data.data() + offset + 22, 4);
        std::memcpy(&fname_len, file_data.data() + offset + 26, 2);
        std::memcpy(&extra_len, file_data.data() + offset + 28, 2);

        std::string fname(file_data.data() + offset + 30, fname_len);

        size_t data_offset = offset + 30 + fname_len + extra_len;

        // Use 64-bit sizes, handling ZIP64 extra field
        uint64_t comp_size = comp_size32;
        uint64_t uncomp_size = uncomp_size32;

        // If sizes are 0xFFFFFFFF, look for ZIP64 extended info in extra field
        if (comp_size32 == 0xFFFFFFFF || uncomp_size32 == 0xFFFFFFFF) {
            const char* extra_ptr = file_data.data() + offset + 30 + fname_len;
            size_t extra_off = 0;
            while (extra_off + 4 <= extra_len) {
                uint16_t hdr_id, hdr_sz;
                std::memcpy(&hdr_id, extra_ptr + extra_off, 2);
                std::memcpy(&hdr_sz, extra_ptr + extra_off + 2, 2);
                if (hdr_id == 0x0001 && hdr_sz >= 16) {
                    // ZIP64 extra field: uncompressed size (8), compressed size (8)
                    std::memcpy(&uncomp_size, extra_ptr + extra_off + 4, 8);
                    std::memcpy(&comp_size, extra_ptr + extra_off + 12, 8);
                    break;
                }
                extra_off += 4 + hdr_sz;
            }
        }

        // Sanity check: skip entries with implausible sizes (corrupt ZIP64)
        if (comp_size > (uint64_t)file_size || uncomp_size > (uint64_t)file_size * 100) {
            offset = data_offset + std::min(comp_size, (uint64_t)(file_size - data_offset));
            continue;
        }

        // Strip .npy extension from name
        std::string key = fname;
        if (key.size() > 4 && key.substr(key.size() - 4) == ".npy") {
            key = key.substr(0, key.size() - 4);
        }

        std::vector<char> npy_data;
        if (compression == 0) {
            // Stored
            npy_data.assign(file_data.data() + data_offset,
                           file_data.data() + data_offset + uncomp_size);
        } else if (compression == 8) {
            // Deflate
            npy_data.resize((size_t)uncomp_size);
            z_stream zs = {};
            zs.next_in = (Bytef*)(file_data.data() + data_offset);
            zs.avail_in = (uInt)comp_size;
            zs.next_out = (Bytef*)npy_data.data();
            zs.avail_out = (uInt)uncomp_size;

            if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
                throw std::runtime_error("inflateInit2 failed for " + fname);
            }
            int ret = inflate(&zs, Z_FINISH);
            inflateEnd(&zs);
            if (ret != Z_STREAM_END) {
                throw std::runtime_error("inflate failed for " + fname);
            }
        } else {
            // Skip unsupported compression
            offset = data_offset + comp_size;
            continue;
        }

        try {
            npz.arrays[key] = parse_npy(npy_data.data(), npy_data.size());
        } catch (const std::exception& e) {
            fprintf(stderr, "  NPZ parse error for '%s': %s\n", key.c_str(), e.what());
        } catch (...) {
            fprintf(stderr, "  NPZ parse error for '%s': unknown\n", key.c_str());
        }

        offset = data_offset + comp_size;
    }

    return npz;
}

} // namespace cls
