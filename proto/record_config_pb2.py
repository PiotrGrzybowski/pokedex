# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: record_config.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='record_config.proto',
  package='pokedex.protos',
  syntax='proto2',
  serialized_pb=_b('\n\x13record_config.proto\x12\x0epokedex.protos\"K\n\x0cRecordConfig\x12\x11\n\tdata_path\x18\x01 \x02(\t\x12\x13\n\x0btarget_size\x18\x02 \x03(\x05\x12\x13\n\x0boutput_path\x18\x03 \x02(\t')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_RECORDCONFIG = _descriptor.Descriptor(
  name='RecordConfig',
  full_name='pokedex.protos.RecordConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data_path', full_name='pokedex.protos.RecordConfig.data_path', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='target_size', full_name='pokedex.protos.RecordConfig.target_size', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='output_path', full_name='pokedex.protos.RecordConfig.output_path', index=2,
      number=3, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=39,
  serialized_end=114,
)

DESCRIPTOR.message_types_by_name['RecordConfig'] = _RECORDCONFIG

RecordConfig = _reflection.GeneratedProtocolMessageType('RecordConfig', (_message.Message,), dict(
  DESCRIPTOR = _RECORDCONFIG,
  __module__ = 'record_config_pb2'
  # @@protoc_insertion_point(class_scope:pokedex.protos.RecordConfig)
  ))
_sym_db.RegisterMessage(RecordConfig)


# @@protoc_insertion_point(module_scope)
