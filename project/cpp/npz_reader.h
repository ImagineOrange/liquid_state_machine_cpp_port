#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace cls {

struct NpyArray {
    std::vector<size_t> shape;
    std::string dtype;  // e.g. "<f8", "<i4", "<i8", "|S6"
    int elem_size = 0;
    std::vector<char> raw_data;

    size_t numel() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }

    // Typed accessors
    const double* as_float64() const { return reinterpret_cast<const double*>(raw_data.data()); }
    const int32_t* as_int32() const { return reinterpret_cast<const int32_t*>(raw_data.data()); }
    const int64_t* as_int64() const { return reinterpret_cast<const int64_t*>(raw_data.data()); }

    int64_t as_scalar_int() const;
    std::string as_string() const;

    std::vector<double> to_float64_vec() const;
    std::vector<int32_t> to_int32_vec() const;
};

struct NpzFile {
    std::map<std::string, NpyArray> arrays;
    bool has(const std::string& name) const { return arrays.count(name) > 0; }
    const NpyArray& operator[](const std::string& name) const { return arrays.at(name); }
};

NpzFile load_npz(const std::string& path);

} // namespace cls
