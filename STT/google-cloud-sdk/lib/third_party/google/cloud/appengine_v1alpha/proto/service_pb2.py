# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: google/cloud/appengine_v1alpha/proto/service.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='google/cloud/appengine_v1alpha/proto/service.proto',
  package='google.appengine.v1alpha',
  syntax='proto3',
  serialized_options=b'\n\034com.google.appengine.v1alphaB\014ServiceProtoP\001ZAgoogle.golang.org/genproto/googleapis/appengine/v1alpha;appengine',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n2google/cloud/appengine_v1alpha/proto/service.proto\x12\x18google.appengine.v1alpha\x1a\x1cgoogle/api/annotations.proto\"Z\n\x07Service\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\t\x12\x35\n\x05split\x18\x03 \x01(\x0b\x32&.google.appengine.v1alpha.TrafficSplit\"\x8e\x02\n\x0cTrafficSplit\x12@\n\x08shard_by\x18\x01 \x01(\x0e\x32..google.appengine.v1alpha.TrafficSplit.ShardBy\x12L\n\x0b\x61llocations\x18\x02 \x03(\x0b\x32\x37.google.appengine.v1alpha.TrafficSplit.AllocationsEntry\x1a\x32\n\x10\x41llocationsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x01:\x02\x38\x01\":\n\x07ShardBy\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\n\n\x06\x43OOKIE\x10\x01\x12\x06\n\x02IP\x10\x02\x12\n\n\x06RANDOM\x10\x03\x42q\n\x1c\x63om.google.appengine.v1alphaB\x0cServiceProtoP\x01ZAgoogle.golang.org/genproto/googleapis/appengine/v1alpha;appengineb\x06proto3'
  ,
  dependencies=[google_dot_api_dot_annotations__pb2.DESCRIPTOR,])



_TRAFFICSPLIT_SHARDBY = _descriptor.EnumDescriptor(
  name='ShardBy',
  full_name='google.appengine.v1alpha.TrafficSplit.ShardBy',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='COOKIE', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='IP', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='RANDOM', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=415,
  serialized_end=473,
)
_sym_db.RegisterEnumDescriptor(_TRAFFICSPLIT_SHARDBY)


_SERVICE = _descriptor.Descriptor(
  name='Service',
  full_name='google.appengine.v1alpha.Service',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='google.appengine.v1alpha.Service.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='id', full_name='google.appengine.v1alpha.Service.id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='split', full_name='google.appengine.v1alpha.Service.split', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=110,
  serialized_end=200,
)


_TRAFFICSPLIT_ALLOCATIONSENTRY = _descriptor.Descriptor(
  name='AllocationsEntry',
  full_name='google.appengine.v1alpha.TrafficSplit.AllocationsEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='google.appengine.v1alpha.TrafficSplit.AllocationsEntry.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='value', full_name='google.appengine.v1alpha.TrafficSplit.AllocationsEntry.value', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=b'8\001',
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=363,
  serialized_end=413,
)

_TRAFFICSPLIT = _descriptor.Descriptor(
  name='TrafficSplit',
  full_name='google.appengine.v1alpha.TrafficSplit',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='shard_by', full_name='google.appengine.v1alpha.TrafficSplit.shard_by', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='allocations', full_name='google.appengine.v1alpha.TrafficSplit.allocations', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_TRAFFICSPLIT_ALLOCATIONSENTRY, ],
  enum_types=[
    _TRAFFICSPLIT_SHARDBY,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=203,
  serialized_end=473,
)

_SERVICE.fields_by_name['split'].message_type = _TRAFFICSPLIT
_TRAFFICSPLIT_ALLOCATIONSENTRY.containing_type = _TRAFFICSPLIT
_TRAFFICSPLIT.fields_by_name['shard_by'].enum_type = _TRAFFICSPLIT_SHARDBY
_TRAFFICSPLIT.fields_by_name['allocations'].message_type = _TRAFFICSPLIT_ALLOCATIONSENTRY
_TRAFFICSPLIT_SHARDBY.containing_type = _TRAFFICSPLIT
DESCRIPTOR.message_types_by_name['Service'] = _SERVICE
DESCRIPTOR.message_types_by_name['TrafficSplit'] = _TRAFFICSPLIT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Service = _reflection.GeneratedProtocolMessageType('Service', (_message.Message,), {
  'DESCRIPTOR' : _SERVICE,
  '__module__' : 'google.cloud.appengine_v1alpha.proto.service_pb2'
  ,
  '__doc__': """A Service resource is a logical component of an application that can
  share state and communicate in a secure fashion with other services.
  For example, an application that handles customer requests might
  include separate services to handle tasks such as backend data
  analysis or API requests from mobile devices. Each service has a
  collection of versions that define a specific set of code used to
  implement the functionality of that service.
  
  Attributes:
      name:
          Full path to the Service resource in the API. Example:
          ``apps/myapp/services/default``.  @OutputOnly
      id:
          Relative name of the service within the application. Example:
          ``default``.  @OutputOnly
      split:
          Mapping that defines fractional HTTP traffic diversion to
          different versions within the service.
  """,
  # @@protoc_insertion_point(class_scope:google.appengine.v1alpha.Service)
  })
_sym_db.RegisterMessage(Service)

TrafficSplit = _reflection.GeneratedProtocolMessageType('TrafficSplit', (_message.Message,), {

  'AllocationsEntry' : _reflection.GeneratedProtocolMessageType('AllocationsEntry', (_message.Message,), {
    'DESCRIPTOR' : _TRAFFICSPLIT_ALLOCATIONSENTRY,
    '__module__' : 'google.cloud.appengine_v1alpha.proto.service_pb2'
    # @@protoc_insertion_point(class_scope:google.appengine.v1alpha.TrafficSplit.AllocationsEntry)
    })
  ,
  'DESCRIPTOR' : _TRAFFICSPLIT,
  '__module__' : 'google.cloud.appengine_v1alpha.proto.service_pb2'
  ,
  '__doc__': """Traffic routing configuration for versions within a single service.
  Traffic splits define how traffic directed to the service is assigned
  to versions.
  
  Attributes:
      shard_by:
          Mechanism used to determine which version a request is sent
          to. The traffic selection algorithm will be stable for either
          type until allocations are changed.
      allocations:
          Mapping from version IDs within the service to fractional
          (0.000, 1] allocations of traffic for that version. Each
          version can be specified only once, but some versions in the
          service may not have any traffic allocation. Services that
          have traffic allocated cannot be deleted until either the
          service is deleted or their traffic allocation is removed.
          Allocations must sum to 1. Up to two decimal place precision
          is supported for IP-based splits and up to three decimal
          places is supported for cookie-based splits.
  """,
  # @@protoc_insertion_point(class_scope:google.appengine.v1alpha.TrafficSplit)
  })
_sym_db.RegisterMessage(TrafficSplit)
_sym_db.RegisterMessage(TrafficSplit.AllocationsEntry)


DESCRIPTOR._options = None
_TRAFFICSPLIT_ALLOCATIONSENTRY._options = None
# @@protoc_insertion_point(module_scope)
