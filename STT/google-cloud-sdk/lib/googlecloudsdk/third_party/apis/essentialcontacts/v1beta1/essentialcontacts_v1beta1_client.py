"""Generated client library for essentialcontacts version v1beta1."""
# NOTE: This file is autogenerated and should not be edited by hand.

from __future__ import absolute_import

from apitools.base.py import base_api
from googlecloudsdk.third_party.apis.essentialcontacts.v1beta1 import essentialcontacts_v1beta1_messages as messages


class EssentialcontactsV1beta1(base_api.BaseApiClient):
  """Generated client library for service essentialcontacts version v1beta1."""

  MESSAGES_MODULE = messages
  BASE_URL = 'https://essentialcontacts.googleapis.com/'
  MTLS_BASE_URL = 'https://essentialcontacts.mtls.googleapis.com/'

  _PACKAGE = 'essentialcontacts'
  _SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
  _VERSION = 'v1beta1'
  _CLIENT_ID = 'CLIENT_ID'
  _CLIENT_SECRET = 'CLIENT_SECRET'
  _USER_AGENT = 'google-cloud-sdk'
  _CLIENT_CLASS_NAME = 'EssentialcontactsV1beta1'
  _URL_VERSION = 'v1beta1'
  _API_KEY = None

  def __init__(self, url='', credentials=None,
               get_credentials=True, http=None, model=None,
               log_request=False, log_response=False,
               credentials_args=None, default_global_params=None,
               additional_http_headers=None, response_encoding=None):
    """Create a new essentialcontacts handle."""
    url = url or self.BASE_URL
    super(EssentialcontactsV1beta1, self).__init__(
        url, credentials=credentials,
        get_credentials=get_credentials, http=http, model=model,
        log_request=log_request, log_response=log_response,
        credentials_args=credentials_args,
        default_global_params=default_global_params,
        additional_http_headers=additional_http_headers,
        response_encoding=response_encoding)
    self.folders_contacts = self.FoldersContactsService(self)
    self.folders = self.FoldersService(self)
    self.organizations_contacts = self.OrganizationsContactsService(self)
    self.organizations = self.OrganizationsService(self)
    self.projects_contacts = self.ProjectsContactsService(self)
    self.projects = self.ProjectsService(self)

  class FoldersContactsService(base_api.BaseApiService):
    """Service class for the folders_contacts resource."""

    _NAME = 'folders_contacts'

    def __init__(self, client):
      super(EssentialcontactsV1beta1.FoldersContactsService, self).__init__(client)
      self._upload_configs = {
          }

    def Compute(self, request, global_params=None):
      r"""Lists all contacts for the resource that are subscribed to the specified notification categories, including contacts inherited from any parent resources.

      Args:
        request: (EssentialcontactsFoldersContactsComputeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1ComputeContactsResponse) The response message.
      """
      config = self.GetMethodConfig('Compute')
      return self._RunMethod(
          config, request, global_params=global_params)

    Compute.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/folders/{foldersId}/contacts:compute',
        http_method='GET',
        method_id='essentialcontacts.folders.contacts.compute',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['notificationCategories', 'pageSize', 'pageToken'],
        relative_path='v1beta1/{+parent}/contacts:compute',
        request_field='',
        request_type_name='EssentialcontactsFoldersContactsComputeRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1ComputeContactsResponse',
        supports_download=False,
    )

    def Create(self, request, global_params=None):
      r"""Adds a new contact for a resource.

      Args:
        request: (EssentialcontactsFoldersContactsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1Contact) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/folders/{foldersId}/contacts',
        http_method='POST',
        method_id='essentialcontacts.folders.contacts.create',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=[],
        relative_path='v1beta1/{+parent}/contacts',
        request_field='googleCloudEssentialcontactsV1beta1Contact',
        request_type_name='EssentialcontactsFoldersContactsCreateRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1Contact',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Deletes a contact.

      Args:
        request: (EssentialcontactsFoldersContactsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/folders/{foldersId}/contacts/{contactsId}',
        http_method='DELETE',
        method_id='essentialcontacts.folders.contacts.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1beta1/{+name}',
        request_field='',
        request_type_name='EssentialcontactsFoldersContactsDeleteRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Gets a single contact.

      Args:
        request: (EssentialcontactsFoldersContactsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1Contact) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/folders/{foldersId}/contacts/{contactsId}',
        http_method='GET',
        method_id='essentialcontacts.folders.contacts.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1beta1/{+name}',
        request_field='',
        request_type_name='EssentialcontactsFoldersContactsGetRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1Contact',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Lists the contacts that have been set on a resource.

      Args:
        request: (EssentialcontactsFoldersContactsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1ListContactsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/folders/{foldersId}/contacts',
        http_method='GET',
        method_id='essentialcontacts.folders.contacts.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['pageSize', 'pageToken'],
        relative_path='v1beta1/{+parent}/contacts',
        request_field='',
        request_type_name='EssentialcontactsFoldersContactsListRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1ListContactsResponse',
        supports_download=False,
    )

    def Patch(self, request, global_params=None):
      r"""Updates a contact. Note: A contact's email address cannot be changed.

      Args:
        request: (EssentialcontactsFoldersContactsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1Contact) The response message.
      """
      config = self.GetMethodConfig('Patch')
      return self._RunMethod(
          config, request, global_params=global_params)

    Patch.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/folders/{foldersId}/contacts/{contactsId}',
        http_method='PATCH',
        method_id='essentialcontacts.folders.contacts.patch',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['updateMask'],
        relative_path='v1beta1/{+name}',
        request_field='googleCloudEssentialcontactsV1beta1Contact',
        request_type_name='EssentialcontactsFoldersContactsPatchRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1Contact',
        supports_download=False,
    )

    def SendTestMessage(self, request, global_params=None):
      r"""Allows a contact admin to send a test message to contact to verify that it has been configured correctly.

      Args:
        request: (EssentialcontactsFoldersContactsSendTestMessageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('SendTestMessage')
      return self._RunMethod(
          config, request, global_params=global_params)

    SendTestMessage.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/folders/{foldersId}/contacts:sendTestMessage',
        http_method='POST',
        method_id='essentialcontacts.folders.contacts.sendTestMessage',
        ordered_params=['resource'],
        path_params=['resource'],
        query_params=[],
        relative_path='v1beta1/{+resource}/contacts:sendTestMessage',
        request_field='googleCloudEssentialcontactsV1beta1SendTestMessageRequest',
        request_type_name='EssentialcontactsFoldersContactsSendTestMessageRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

  class FoldersService(base_api.BaseApiService):
    """Service class for the folders resource."""

    _NAME = 'folders'

    def __init__(self, client):
      super(EssentialcontactsV1beta1.FoldersService, self).__init__(client)
      self._upload_configs = {
          }

  class OrganizationsContactsService(base_api.BaseApiService):
    """Service class for the organizations_contacts resource."""

    _NAME = 'organizations_contacts'

    def __init__(self, client):
      super(EssentialcontactsV1beta1.OrganizationsContactsService, self).__init__(client)
      self._upload_configs = {
          }

    def Compute(self, request, global_params=None):
      r"""Lists all contacts for the resource that are subscribed to the specified notification categories, including contacts inherited from any parent resources.

      Args:
        request: (EssentialcontactsOrganizationsContactsComputeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1ComputeContactsResponse) The response message.
      """
      config = self.GetMethodConfig('Compute')
      return self._RunMethod(
          config, request, global_params=global_params)

    Compute.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/organizations/{organizationsId}/contacts:compute',
        http_method='GET',
        method_id='essentialcontacts.organizations.contacts.compute',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['notificationCategories', 'pageSize', 'pageToken'],
        relative_path='v1beta1/{+parent}/contacts:compute',
        request_field='',
        request_type_name='EssentialcontactsOrganizationsContactsComputeRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1ComputeContactsResponse',
        supports_download=False,
    )

    def Create(self, request, global_params=None):
      r"""Adds a new contact for a resource.

      Args:
        request: (EssentialcontactsOrganizationsContactsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1Contact) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/organizations/{organizationsId}/contacts',
        http_method='POST',
        method_id='essentialcontacts.organizations.contacts.create',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=[],
        relative_path='v1beta1/{+parent}/contacts',
        request_field='googleCloudEssentialcontactsV1beta1Contact',
        request_type_name='EssentialcontactsOrganizationsContactsCreateRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1Contact',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Deletes a contact.

      Args:
        request: (EssentialcontactsOrganizationsContactsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/organizations/{organizationsId}/contacts/{contactsId}',
        http_method='DELETE',
        method_id='essentialcontacts.organizations.contacts.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1beta1/{+name}',
        request_field='',
        request_type_name='EssentialcontactsOrganizationsContactsDeleteRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Gets a single contact.

      Args:
        request: (EssentialcontactsOrganizationsContactsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1Contact) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/organizations/{organizationsId}/contacts/{contactsId}',
        http_method='GET',
        method_id='essentialcontacts.organizations.contacts.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1beta1/{+name}',
        request_field='',
        request_type_name='EssentialcontactsOrganizationsContactsGetRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1Contact',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Lists the contacts that have been set on a resource.

      Args:
        request: (EssentialcontactsOrganizationsContactsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1ListContactsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/organizations/{organizationsId}/contacts',
        http_method='GET',
        method_id='essentialcontacts.organizations.contacts.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['pageSize', 'pageToken'],
        relative_path='v1beta1/{+parent}/contacts',
        request_field='',
        request_type_name='EssentialcontactsOrganizationsContactsListRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1ListContactsResponse',
        supports_download=False,
    )

    def Patch(self, request, global_params=None):
      r"""Updates a contact. Note: A contact's email address cannot be changed.

      Args:
        request: (EssentialcontactsOrganizationsContactsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1Contact) The response message.
      """
      config = self.GetMethodConfig('Patch')
      return self._RunMethod(
          config, request, global_params=global_params)

    Patch.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/organizations/{organizationsId}/contacts/{contactsId}',
        http_method='PATCH',
        method_id='essentialcontacts.organizations.contacts.patch',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['updateMask'],
        relative_path='v1beta1/{+name}',
        request_field='googleCloudEssentialcontactsV1beta1Contact',
        request_type_name='EssentialcontactsOrganizationsContactsPatchRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1Contact',
        supports_download=False,
    )

    def SendTestMessage(self, request, global_params=None):
      r"""Allows a contact admin to send a test message to contact to verify that it has been configured correctly.

      Args:
        request: (EssentialcontactsOrganizationsContactsSendTestMessageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('SendTestMessage')
      return self._RunMethod(
          config, request, global_params=global_params)

    SendTestMessage.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/organizations/{organizationsId}/contacts:sendTestMessage',
        http_method='POST',
        method_id='essentialcontacts.organizations.contacts.sendTestMessage',
        ordered_params=['resource'],
        path_params=['resource'],
        query_params=[],
        relative_path='v1beta1/{+resource}/contacts:sendTestMessage',
        request_field='googleCloudEssentialcontactsV1beta1SendTestMessageRequest',
        request_type_name='EssentialcontactsOrganizationsContactsSendTestMessageRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

  class OrganizationsService(base_api.BaseApiService):
    """Service class for the organizations resource."""

    _NAME = 'organizations'

    def __init__(self, client):
      super(EssentialcontactsV1beta1.OrganizationsService, self).__init__(client)
      self._upload_configs = {
          }

  class ProjectsContactsService(base_api.BaseApiService):
    """Service class for the projects_contacts resource."""

    _NAME = 'projects_contacts'

    def __init__(self, client):
      super(EssentialcontactsV1beta1.ProjectsContactsService, self).__init__(client)
      self._upload_configs = {
          }

    def Compute(self, request, global_params=None):
      r"""Lists all contacts for the resource that are subscribed to the specified notification categories, including contacts inherited from any parent resources.

      Args:
        request: (EssentialcontactsProjectsContactsComputeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1ComputeContactsResponse) The response message.
      """
      config = self.GetMethodConfig('Compute')
      return self._RunMethod(
          config, request, global_params=global_params)

    Compute.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/projects/{projectsId}/contacts:compute',
        http_method='GET',
        method_id='essentialcontacts.projects.contacts.compute',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['notificationCategories', 'pageSize', 'pageToken'],
        relative_path='v1beta1/{+parent}/contacts:compute',
        request_field='',
        request_type_name='EssentialcontactsProjectsContactsComputeRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1ComputeContactsResponse',
        supports_download=False,
    )

    def Create(self, request, global_params=None):
      r"""Adds a new contact for a resource.

      Args:
        request: (EssentialcontactsProjectsContactsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1Contact) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/projects/{projectsId}/contacts',
        http_method='POST',
        method_id='essentialcontacts.projects.contacts.create',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=[],
        relative_path='v1beta1/{+parent}/contacts',
        request_field='googleCloudEssentialcontactsV1beta1Contact',
        request_type_name='EssentialcontactsProjectsContactsCreateRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1Contact',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      r"""Deletes a contact.

      Args:
        request: (EssentialcontactsProjectsContactsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/projects/{projectsId}/contacts/{contactsId}',
        http_method='DELETE',
        method_id='essentialcontacts.projects.contacts.delete',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1beta1/{+name}',
        request_field='',
        request_type_name='EssentialcontactsProjectsContactsDeleteRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      r"""Gets a single contact.

      Args:
        request: (EssentialcontactsProjectsContactsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1Contact) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/projects/{projectsId}/contacts/{contactsId}',
        http_method='GET',
        method_id='essentialcontacts.projects.contacts.get',
        ordered_params=['name'],
        path_params=['name'],
        query_params=[],
        relative_path='v1beta1/{+name}',
        request_field='',
        request_type_name='EssentialcontactsProjectsContactsGetRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1Contact',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      r"""Lists the contacts that have been set on a resource.

      Args:
        request: (EssentialcontactsProjectsContactsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1ListContactsResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/projects/{projectsId}/contacts',
        http_method='GET',
        method_id='essentialcontacts.projects.contacts.list',
        ordered_params=['parent'],
        path_params=['parent'],
        query_params=['pageSize', 'pageToken'],
        relative_path='v1beta1/{+parent}/contacts',
        request_field='',
        request_type_name='EssentialcontactsProjectsContactsListRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1ListContactsResponse',
        supports_download=False,
    )

    def Patch(self, request, global_params=None):
      r"""Updates a contact. Note: A contact's email address cannot be changed.

      Args:
        request: (EssentialcontactsProjectsContactsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudEssentialcontactsV1beta1Contact) The response message.
      """
      config = self.GetMethodConfig('Patch')
      return self._RunMethod(
          config, request, global_params=global_params)

    Patch.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/projects/{projectsId}/contacts/{contactsId}',
        http_method='PATCH',
        method_id='essentialcontacts.projects.contacts.patch',
        ordered_params=['name'],
        path_params=['name'],
        query_params=['updateMask'],
        relative_path='v1beta1/{+name}',
        request_field='googleCloudEssentialcontactsV1beta1Contact',
        request_type_name='EssentialcontactsProjectsContactsPatchRequest',
        response_type_name='GoogleCloudEssentialcontactsV1beta1Contact',
        supports_download=False,
    )

    def SendTestMessage(self, request, global_params=None):
      r"""Allows a contact admin to send a test message to contact to verify that it has been configured correctly.

      Args:
        request: (EssentialcontactsProjectsContactsSendTestMessageRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
      config = self.GetMethodConfig('SendTestMessage')
      return self._RunMethod(
          config, request, global_params=global_params)

    SendTestMessage.method_config = lambda: base_api.ApiMethodInfo(
        flat_path='v1beta1/projects/{projectsId}/contacts:sendTestMessage',
        http_method='POST',
        method_id='essentialcontacts.projects.contacts.sendTestMessage',
        ordered_params=['resource'],
        path_params=['resource'],
        query_params=[],
        relative_path='v1beta1/{+resource}/contacts:sendTestMessage',
        request_field='googleCloudEssentialcontactsV1beta1SendTestMessageRequest',
        request_type_name='EssentialcontactsProjectsContactsSendTestMessageRequest',
        response_type_name='GoogleProtobufEmpty',
        supports_download=False,
    )

  class ProjectsService(base_api.BaseApiService):
    """Service class for the projects resource."""

    _NAME = 'projects'

    def __init__(self, client):
      super(EssentialcontactsV1beta1.ProjectsService, self).__init__(client)
      self._upload_configs = {
          }
