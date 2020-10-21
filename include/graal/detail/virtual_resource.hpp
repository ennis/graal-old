#pragma once
#include <graal/detail/image_resource.hpp>
#include <graal/detail/named_object.hpp>
#include <graal/image_format.hpp>
#include <graal/image_type.hpp>
#include <memory>

namespace graal::detail {

using temporary_index = std::size_t;
constexpr temporary_index invalid_temporary_index = (temporary_index)-1;

/// @brief
class virtual_resource : public named_object {
  friend class queue_impl;

public:
  virtual_resource() {}

  void discard() { discarded_ = true; }

  /// @brief Returns whether the memory for this resource can be shared with the
  /// other virtual resource.
  /// @param other
  /// @return
  virtual bool is_aliasable_with(const virtual_resource &other) const = 0;

  /// @brief Allocates a concrete resource.
  virtual void allocate() = 0;
  
  /// @brief 
  /// @param other 
  virtual void alias_with(virtual_resource& other) = 0;

protected:
  bool discarded_ = false; // no more external references to this resource
  bool noalias_ = false;    // explicitly forbid aliasing
  temporary_index tmp_index_ =
      invalid_temporary_index; // assigned temporary index
};

/// @brief
class virtual_image_resource final : public virtual_resource {
  friend class queue_impl;

public:
  virtual_image_resource(const image_desc &desc) : desc_{desc} {}

  bool is_aliasable_with(const virtual_resource &other) const override;
  void allocate() override;
  void alias_with(virtual_resource& other) override;

private:
  image_desc                      desc_;
  std::shared_ptr<image_resource> image_;
};

/// @brief
class virtual_buffer_resource : public virtual_resource {
  friend class queue_impl;

public:
  virtual_buffer_resource(size_t size) : size_{size} {}

  bool is_aliasable_with(const virtual_resource &other) const override;
  void allocate() override;
  void alias_with(virtual_resource& other) override;


private:
  size_t size_;
};

using virtual_resource_ptr = std::shared_ptr<virtual_resource>;
} // namespace graal::detail