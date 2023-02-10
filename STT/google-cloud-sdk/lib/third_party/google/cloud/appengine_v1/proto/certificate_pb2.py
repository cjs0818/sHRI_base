# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: google/cloud/appengine_v1/proto/certificate.proto

from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.api import annotations_pb2 as google_dot_api_dot_annotations__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='google/cloud/appengine_v1/proto/certificate.proto',
  package='google.appengine.v1',
  syntax='proto3',
  serialized_options=b'\n\027com.google.appengine.v1B\020CertificateProtoP\001Z<google.golang.org/genproto/googleapis/appengine/v1;appengine',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n1google/cloud/appengine_v1/proto/certificate.proto\x12\x13google.appengine.v1\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1cgoogle/api/annotations.proto\"\xdb\x02\n\x15\x41uthorizedCertificate\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\n\n\x02id\x18\x02 \x01(\t\x12\x14\n\x0c\x64isplay_name\x18\x03 \x01(\t\x12\x14\n\x0c\x64omain_names\x18\x04 \x03(\t\x12/\n\x0b\x65xpire_time\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x45\n\x14\x63\x65rtificate_raw_data\x18\x06 \x01(\x0b\x32\'.google.appengine.v1.CertificateRawData\x12\x44\n\x13managed_certificate\x18\x07 \x01(\x0b\x32\'.google.appengine.v1.ManagedCertificate\x12\x1f\n\x17visible_domain_mappings\x18\x08 \x03(\t\x12\x1d\n\x15\x64omain_mappings_count\x18\t \x01(\x05\"E\n\x12\x43\x65rtificateRawData\x12\x1a\n\x12public_certificate\x18\x01 \x01(\t\x12\x13\n\x0bprivate_key\x18\x02 \x01(\t\"\x82\x01\n\x12ManagedCertificate\x12\x35\n\x11last_renewal_time\x18\x01 \x01(\x0b\x32\x1a.google.protobuf.Timestamp\x12\x35\n\x06status\x18\x02 \x01(\x0e\x32%.google.appengine.v1.ManagementStatus*\xc6\x01\n\x10ManagementStatus\x12!\n\x1dMANAGEMENT_STATUS_UNSPECIFIED\x10\x00\x12\x06\n\x02OK\x10\x01\x12\x0b\n\x07PENDING\x10\x02\x12\x1f\n\x1b\x46\x41ILED_RETRYING_NOT_VISIBLE\x10\x04\x12\x14\n\x10\x46\x41ILED_PERMANENT\x10\x06\x12!\n\x1d\x46\x41ILED_RETRYING_CAA_FORBIDDEN\x10\x07\x12 \n\x1c\x46\x41ILED_RETRYING_CAA_CHECKING\x10\x08\x42k\n\x17\x63om.google.appengine.v1B\x10\x43\x65rtificateProtoP\x01Z<google.golang.org/genproto/googleapis/appengine/v1;appengineb\x06proto3'
  ,
  dependencies=[google_dot_protobuf_dot_timestamp__pb2.DESCRIPTOR,google_dot_api_dot_annotations__pb2.DESCRIPTOR,])

_MANAGEMENTSTATUS = _descriptor.EnumDescriptor(
  name='ManagementStatus',
  full_name='google.appengine.v1.ManagementStatus',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='MANAGEMENT_STATUS_UNSPECIFIED', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='OK', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PENDING', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FAILED_RETRYING_NOT_VISIBLE', index=3, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FAILED_PERMANENT', index=4, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FAILED_RETRYING_CAA_FORBIDDEN', index=5, number=7,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FAILED_RETRYING_CAA_CHECKING', index=6, number=8,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=692,
  serialized_end=890,
)
_sym_db.RegisterEnumDescriptor(_MANAGEMENTSTATUS)

ManagementStatus = enum_type_wrapper.EnumTypeWrapper(_MANAGEMENTSTATUS)
MANAGEMENT_STATUS_UNSPECIFIED = 0
OK = 1
PENDING = 2
FAILED_RETRYING_NOT_VISIBLE = 4
FAILED_PERMANENT = 6
FAILED_RETRYING_CAA_FORBIDDEN = 7
FAILED_RETRYING_CAA_CHECKING = 8



_AUTHORIZEDCERTIFICATE = _descriptor.Descriptor(
  name='AuthorizedCertificate',
  full_name='google.appengine.v1.AuthorizedCertificate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='google.appengine.v1.AuthorizedCertificate.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='id', full_name='google.appengine.v1.AuthorizedCertificate.id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='display_name', full_name='google.appengine.v1.AuthorizedCertificate.display_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='domain_names', full_name='google.appengine.v1.AuthorizedCertificate.domain_names', index=3,
      number=4, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='expire_time', full_name='google.appengine.v1.AuthorizedCertificate.expire_time', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='certificate_raw_data', full_name='google.appengine.v1.AuthorizedCertificate.certificate_raw_data', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='managed_certificate', full_name='google.appengine.v1.AuthorizedCertificate.managed_certificate', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='visible_domain_mappings', full_name='google.appengine.v1.AuthorizedCertificate.visible_domain_mappings', index=7,
      number=8, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='domain_mappings_count', full_name='google.appengine.v1.AuthorizedCertificate.domain_mappings_count', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=138,
  serialized_end=485,
)


_CERTIFICATERAWDATA = _descriptor.Descriptor(
  name='CertificateRawData',
  full_name='google.appengine.v1.CertificateRawData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='public_certificate', full_name='google.appengine.v1.CertificateRawData.public_certificate', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='private_key', full_name='google.appengine.v1.CertificateRawData.private_key', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
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
  serialized_start=487,
  serialized_end=556,
)


_MANAGEDCERTIFICATE = _descriptor.Descriptor(
  name='ManagedCertificate',
  full_name='google.appengine.v1.ManagedCertificate',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='last_renewal_time', full_name='google.appengine.v1.ManagedCertificate.last_renewal_time', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='status', full_name='google.appengine.v1.ManagedCertificate.status', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
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
  serialized_start=559,
  serialized_end=689,
)

_AUTHORIZEDCERTIFICATE.fields_by_name['expire_time'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_AUTHORIZEDCERTIFICATE.fields_by_name['certificate_raw_data'].message_type = _CERTIFICATERAWDATA
_AUTHORIZEDCERTIFICATE.fields_by_name['managed_certificate'].message_type = _MANAGEDCERTIFICATE
_MANAGEDCERTIFICATE.fields_by_name['last_renewal_time'].message_type = google_dot_protobuf_dot_timestamp__pb2._TIMESTAMP
_MANAGEDCERTIFICATE.fields_by_name['status'].enum_type = _MANAGEMENTSTATUS
DESCRIPTOR.message_types_by_name['AuthorizedCertificate'] = _AUTHORIZEDCERTIFICATE
DESCRIPTOR.message_types_by_name['CertificateRawData'] = _CERTIFICATERAWDATA
DESCRIPTOR.message_types_by_name['ManagedCertificate'] = _MANAGEDCERTIFICATE
DESCRIPTOR.enum_types_by_name['ManagementStatus'] = _MANAGEMENTSTATUS
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AuthorizedCertificate = _reflection.GeneratedProtocolMessageType('AuthorizedCertificate', (_message.Message,), {
  'DESCRIPTOR' : _AUTHORIZEDCERTIFICATE,
  '__module__' : 'google.cloud.appengine_v1.proto.certificate_pb2'
  ,
  '__doc__': """An SSL certificate that a user has been authorized to administer. A
  user is authorized to administer any certificate that applies to one
  of their authorized domains.
  
  Attributes:
      name:
          Full path to the ``AuthorizedCertificate`` resource in the
          API. Example: ``apps/myapp/authorizedCertificates/12345``.
          @OutputOnly
      id:
          Relative name of the certificate. This is a unique value
          autogenerated on ``AuthorizedCertificate`` resource creation.
          Example: ``12345``.  @OutputOnly
      display_name:
          The user-specified display name of the certificate. This is
          not guaranteed to be unique. Example: ``My Certificate``.
      domain_names:
          Topmost applicable domains of this certificate. This
          certificate applies to these domains and their subdomains.
          Example: ``example.com``.  @OutputOnly
      expire_time:
          The time when this certificate expires. To update the renewal
          time on this certificate, upload an SSL certificate with a
          different expiration time using
          ```AuthorizedCertificates.UpdateAuthorizedCertificate`` <>`__.
          @OutputOnly
      certificate_raw_data:
          The SSL certificate serving the ``AuthorizedCertificate``
          resource. This must be obtained independently from a
          certificate authority.
      managed_certificate:
          Only applicable if this certificate is managed by App Engine.
          Managed certificates are tied to the lifecycle of a
          ``DomainMapping`` and cannot be updated or deleted via the
          ``AuthorizedCertificates`` API. If this certificate is
          manually administered by the user, this field will be empty.
          @OutputOnly
      visible_domain_mappings:
          The full paths to user visible Domain Mapping resources that
          have this certificate mapped. Example:
          ``apps/myapp/domainMappings/example.com``.  This may not
          represent the full list of mapped domain mappings if the user
          does not have ``VIEWER`` permissions on all of the
          applications that have this certificate mapped. See
          ``domain_mappings_count`` for a complete count.  Only returned
          by ``GET`` or ``LIST`` requests when specifically requested by
          the ``view=FULL_CERTIFICATE`` option.  @OutputOnly
      domain_mappings_count:
          Aggregate count of the domain mappings with this certificate
          mapped. This count includes domain mappings on applications
          for which the user does not have ``VIEWER`` permissions.  Only
          returned by ``GET`` or ``LIST`` requests when specifically
          requested by the ``view=FULL_CERTIFICATE`` option.
          @OutputOnly
  """,
  # @@protoc_insertion_point(class_scope:google.appengine.v1.AuthorizedCertificate)
  })
_sym_db.RegisterMessage(AuthorizedCertificate)

CertificateRawData = _reflection.GeneratedProtocolMessageType('CertificateRawData', (_message.Message,), {
  'DESCRIPTOR' : _CERTIFICATERAWDATA,
  '__module__' : 'google.cloud.appengine_v1.proto.certificate_pb2'
  ,
  '__doc__': """An SSL certificate obtained from a certificate authority.
  
  Attributes:
      public_certificate:
          PEM encoded x.509 public key certificate. This field is set
          once on certificate creation. Must include the header and
          footer. Example:  .. raw:: html     <pre>    -----BEGIN
          CERTIFICATE-----    <certificate_value>    -----END
          CERTIFICATE-----    </pre>
      private_key:
          Unencrypted PEM encoded RSA private key. This field is set
          once on certificate creation and then encrypted. The key size
          must be 2048 bits or fewer. Must include the header and
          footer. Example:  .. raw:: html     <pre>    -----BEGIN RSA
          PRIVATE KEY-----    <unencrypted_key_value>    -----END RSA
          PRIVATE KEY-----    </pre>  @InputOnly
  """,
  # @@protoc_insertion_point(class_scope:google.appengine.v1.CertificateRawData)
  })
_sym_db.RegisterMessage(CertificateRawData)

ManagedCertificate = _reflection.GeneratedProtocolMessageType('ManagedCertificate', (_message.Message,), {
  'DESCRIPTOR' : _MANAGEDCERTIFICATE,
  '__module__' : 'google.cloud.appengine_v1.proto.certificate_pb2'
  ,
  '__doc__': """A certificate managed by App Engine.
  
  Attributes:
      last_renewal_time:
          Time at which the certificate was last renewed. The renewal
          process is fully managed. Certificate renewal will
          automatically occur before the certificate expires. Renewal
          errors can be tracked via ``ManagementStatus``.  @OutputOnly
      status:
          Status of certificate management. Refers to the most recent
          certificate acquisition or renewal attempt.  @OutputOnly
  """,
  # @@protoc_insertion_point(class_scope:google.appengine.v1.ManagedCertificate)
  })
_sym_db.RegisterMessage(ManagedCertificate)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
