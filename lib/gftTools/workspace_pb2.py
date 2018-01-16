# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: workspace.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lib.gftTools import common_pb2
from lib.gftTools import uitypes_pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='workspace.proto',
  package='com.gftchina.common.persistence.workspace',
  serialized_pb=_b('\n\x0fworkspace.proto\x12)com.gftchina.common.persistence.workspace\x1a\x0c\x63ommon.proto\x1a\ruitypes.proto\"\xcd\x02\n\x0cViewSettings\x12]\n\tview_mode\x18\x01 \x01(\x0e\x32@.com.gftchina.common.persistence.workspace.ViewSettings.ViewMode:\x08\x45\x64itMode\x12$\n\x15\x63ompact_mode_dockable\x18\x02 \x01(\x08:\x05\x66\x61lse\x12!\n\x12\x65\x64it_mode_dockable\x18\x03 \x01(\x08:\x05\x66\x61lse\x12\x1a\n\x0ctabs_visible\x18\x04 \x01(\x08:\x04true\x12N\n\rmode_settings\x18\x05 \x01(\x0b\x32\x37.com.gftchina.common.persistence.common.GFTSerializable\")\n\x08ViewMode\x12\x0c\n\x08\x45\x64itMode\x10\x00\x12\x0f\n\x0b\x43ompactMode\x10\x01\"\x94\x03\n\nIOEdgeInfo\x12\x0b\n\x03gid\x18\x01 \x01(\t\x12U\n\x05index\x18\x02 \x01(\x0e\x32\x41.com.gftchina.common.persistence.workspace.IOEdgeInfo.OutputIndex:\x03\x41ll\x12\x11\n\tsource_ui\x18\x03 \x01(\t\x12\x14\n\x0csource_range\x18\x04 \x01(\t\x12\x11\n\ttarget_ui\x18\x05 \x01(\t\x12\x14\n\x0ctarget_range\x18\x06 \x01(\t\x12Z\n\rtarget_action\x18\x07 \x01(\x0e\x32\x43.com.gftchina.common.persistence.workspace.IOEdgeInfo.TargetActions\"0\n\x0bOutputIndex\x12\x07\n\x03\x41ll\x10\x01\x12\r\n\tSelection\x10\x02\x12\t\n\x05\x46ocus\x10\x03\"B\n\rTargetActions\x12\x0f\n\x0b\x41uto_Select\x10\x02\x12\x0f\n\x0b\x41ppend_data\x10\x03\x12\x0f\n\x0bModify_only\x10\x04\"D\n\x0bNewNodeInfo\x12\x0c\n\x04type\x18\x01 \x02(\x05\x12\x14\n\x0csys_sub_type\x18\x02 \x01(\t\x12\x11\n\tname_hint\x18\x03 \x01(\t\"\x89\x06\n\x06UIData\x12@\n\x07ui_type\x18\x01 \x02(\x0e\x32/.com.gftchina.common.persistence.uitypes.UIType\x12\x0b\n\x03gid\x18\x02 \x02(\t\x12\x0c\n\x04name\x18\x03 \x02(\t\x12N\n\rview_settings\x18\x04 \x01(\x0b\x32\x37.com.gftchina.common.persistence.workspace.ViewSettings\x12J\n\tserialize\x18\x05 \x01(\x0b\x32\x37.com.gftchina.common.persistence.common.GFTSerializable\x12\x18\n\x10\x66unctional_block\x18\x06 \x01(\t\x12W\n\x0e\x63ommon_setting\x18\x07 \x01(\x0b\x32?.com.gftchina.common.persistence.workspace.UIData.CommonSetting\x12\x0c\n\x04\x62ody\x18\x08 \x01(\t\x1a\x84\x03\n\rCommonSetting\x12\x1b\n\x13\x61uto_show_skill_bar\x18\x01 \x01(\x08\x12q\n\x0c\x64isp_pattern\x18\x02 \x01(\x0e\x32N.com.gftchina.common.persistence.workspace.UIData.CommonSetting.DisplayPattern:\x0bIconAndHint\x12K\n\x15tool_bar_button_names\x18\x03 \x01(\x0b\x32,.com.gftchina.common.persistence.common.Meta\x12W\n\x17\x61vailable_new_node_type\x18\x04 \x03(\x0b\x32\x36.com.gftchina.common.persistence.workspace.NewNodeInfo\"=\n\x0e\x44isplayPattern\x12\x0f\n\x0bIconAndHint\x10\x00\x12\x0c\n\x08IconOnly\x10\x01\x12\x0c\n\x08HintOnly\x10\x02\"\xdf\x02\n\x14WorkspaceDataOptions\x12H\n\x07layouts\x18\x01 \x01(\x0b\x32\x37.com.gftchina.common.persistence.common.GFTSerializable\x12\x11\n\thide_gids\x18\x02 \x01(\t\x12 \n\x18use_ancestor_publication\x18\x03 \x01(\x08\x12,\n$tryToFindInnerReadonlyDocWhenOpenUrl\x18\x04 \x01(\x08\x12?\n\tshortcuts\x18\x05 \x01(\x0b\x32,.com.gftchina.common.persistence.common.Meta\x12 \n\x18\x63urrent_functional_block\x18\x06 \x01(\t\x12\x1b\n\x13\x64\x65pendent_node_gids\x18\x07 \x03(\t\x12\x1a\n\x12\x61\x63tive_publication\x18\x08 \x01(\t\"\xc4\x02\n\rWorkspaceInfo\x12\x42\n\x07ui_data\x18\x01 \x03(\x0b\x32\x31.com.gftchina.common.persistence.workspace.UIData\x12K\n\x0cio_edge_info\x18\x02 \x03(\x0b\x32\x35.com.gftchina.common.persistence.workspace.IOEdgeInfo\x12O\n\x06ws_opt\x18\x03 \x01(\x0b\x32?.com.gftchina.common.persistence.workspace.WorkspaceDataOptions\x12Q\n\x10ws_view_settings\x18\x04 \x01(\x0b\x32\x37.com.gftchina.common.persistence.workspace.ViewSettings\"\xea\x01\n\x16ServerInterestedWSInfo\x12\x0e\n\x06ws_gid\x18\x01 \x02(\t\x12\x0f\n\x07ws_name\x18\x02 \x02(\t\x12\x1b\n\x13\x63lient_ws_info_data\x18\x03 \x02(\x0c\x12\x12\n\nversion_id\x18\x04 \x01(\x03\x12\x1c\n\x14\x64\x65pendent_nodes_json\x18\x05 \x01(\t\x12\x1b\n\x13\x64\x65pendent_node_gids\x18\x06 \x03(\t\x12\x43\n\rws_properties\x18\x07 \x01(\x0b\x32,.com.gftchina.common.persistence.common.Meta')
  ,
  dependencies=[common_pb2.DESCRIPTOR,uitypes_pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_VIEWSETTINGS_VIEWMODE = _descriptor.EnumDescriptor(
  name='ViewMode',
  full_name='com.gftchina.common.persistence.workspace.ViewSettings.ViewMode',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='EditMode', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CompactMode', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=384,
  serialized_end=425,
)
_sym_db.RegisterEnumDescriptor(_VIEWSETTINGS_VIEWMODE)

_IOEDGEINFO_OUTPUTINDEX = _descriptor.EnumDescriptor(
  name='OutputIndex',
  full_name='com.gftchina.common.persistence.workspace.IOEdgeInfo.OutputIndex',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='All', index=0, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Selection', index=1, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Focus', index=2, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=716,
  serialized_end=764,
)
_sym_db.RegisterEnumDescriptor(_IOEDGEINFO_OUTPUTINDEX)

_IOEDGEINFO_TARGETACTIONS = _descriptor.EnumDescriptor(
  name='TargetActions',
  full_name='com.gftchina.common.persistence.workspace.IOEdgeInfo.TargetActions',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='Auto_Select', index=0, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Append_data', index=1, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='Modify_only', index=2, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=766,
  serialized_end=832,
)
_sym_db.RegisterEnumDescriptor(_IOEDGEINFO_TARGETACTIONS)

_UIDATA_COMMONSETTING_DISPLAYPATTERN = _descriptor.EnumDescriptor(
  name='DisplayPattern',
  full_name='com.gftchina.common.persistence.workspace.UIData.CommonSetting.DisplayPattern',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='IconAndHint', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IconOnly', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HintOnly', index=2, number=2,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1621,
  serialized_end=1682,
)
_sym_db.RegisterEnumDescriptor(_UIDATA_COMMONSETTING_DISPLAYPATTERN)


_VIEWSETTINGS = _descriptor.Descriptor(
  name='ViewSettings',
  full_name='com.gftchina.common.persistence.workspace.ViewSettings',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='view_mode', full_name='com.gftchina.common.persistence.workspace.ViewSettings.view_mode', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='compact_mode_dockable', full_name='com.gftchina.common.persistence.workspace.ViewSettings.compact_mode_dockable', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='edit_mode_dockable', full_name='com.gftchina.common.persistence.workspace.ViewSettings.edit_mode_dockable', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tabs_visible', full_name='com.gftchina.common.persistence.workspace.ViewSettings.tabs_visible', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mode_settings', full_name='com.gftchina.common.persistence.workspace.ViewSettings.mode_settings', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _VIEWSETTINGS_VIEWMODE,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=92,
  serialized_end=425,
)


_IOEDGEINFO = _descriptor.Descriptor(
  name='IOEdgeInfo',
  full_name='com.gftchina.common.persistence.workspace.IOEdgeInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='gid', full_name='com.gftchina.common.persistence.workspace.IOEdgeInfo.gid', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='index', full_name='com.gftchina.common.persistence.workspace.IOEdgeInfo.index', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='source_ui', full_name='com.gftchina.common.persistence.workspace.IOEdgeInfo.source_ui', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='source_range', full_name='com.gftchina.common.persistence.workspace.IOEdgeInfo.source_range', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='target_ui', full_name='com.gftchina.common.persistence.workspace.IOEdgeInfo.target_ui', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='target_range', full_name='com.gftchina.common.persistence.workspace.IOEdgeInfo.target_range', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='target_action', full_name='com.gftchina.common.persistence.workspace.IOEdgeInfo.target_action', index=6,
      number=7, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _IOEDGEINFO_OUTPUTINDEX,
    _IOEDGEINFO_TARGETACTIONS,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=428,
  serialized_end=832,
)


_NEWNODEINFO = _descriptor.Descriptor(
  name='NewNodeInfo',
  full_name='com.gftchina.common.persistence.workspace.NewNodeInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='com.gftchina.common.persistence.workspace.NewNodeInfo.type', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sys_sub_type', full_name='com.gftchina.common.persistence.workspace.NewNodeInfo.sys_sub_type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='name_hint', full_name='com.gftchina.common.persistence.workspace.NewNodeInfo.name_hint', index=2,
      number=3, type=9, cpp_type=9, label=1,
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
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=834,
  serialized_end=902,
)


_UIDATA_COMMONSETTING = _descriptor.Descriptor(
  name='CommonSetting',
  full_name='com.gftchina.common.persistence.workspace.UIData.CommonSetting',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='auto_show_skill_bar', full_name='com.gftchina.common.persistence.workspace.UIData.CommonSetting.auto_show_skill_bar', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='disp_pattern', full_name='com.gftchina.common.persistence.workspace.UIData.CommonSetting.disp_pattern', index=1,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=True, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tool_bar_button_names', full_name='com.gftchina.common.persistence.workspace.UIData.CommonSetting.tool_bar_button_names', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='available_new_node_type', full_name='com.gftchina.common.persistence.workspace.UIData.CommonSetting.available_new_node_type', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _UIDATA_COMMONSETTING_DISPLAYPATTERN,
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1294,
  serialized_end=1682,
)

_UIDATA = _descriptor.Descriptor(
  name='UIData',
  full_name='com.gftchina.common.persistence.workspace.UIData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ui_type', full_name='com.gftchina.common.persistence.workspace.UIData.ui_type', index=0,
      number=1, type=14, cpp_type=8, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='gid', full_name='com.gftchina.common.persistence.workspace.UIData.gid', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='name', full_name='com.gftchina.common.persistence.workspace.UIData.name', index=2,
      number=3, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='view_settings', full_name='com.gftchina.common.persistence.workspace.UIData.view_settings', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='serialize', full_name='com.gftchina.common.persistence.workspace.UIData.serialize', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='functional_block', full_name='com.gftchina.common.persistence.workspace.UIData.functional_block', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='common_setting', full_name='com.gftchina.common.persistence.workspace.UIData.common_setting', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='body', full_name='com.gftchina.common.persistence.workspace.UIData.body', index=7,
      number=8, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_UIDATA_COMMONSETTING, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=905,
  serialized_end=1682,
)


_WORKSPACEDATAOPTIONS = _descriptor.Descriptor(
  name='WorkspaceDataOptions',
  full_name='com.gftchina.common.persistence.workspace.WorkspaceDataOptions',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='layouts', full_name='com.gftchina.common.persistence.workspace.WorkspaceDataOptions.layouts', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hide_gids', full_name='com.gftchina.common.persistence.workspace.WorkspaceDataOptions.hide_gids', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='use_ancestor_publication', full_name='com.gftchina.common.persistence.workspace.WorkspaceDataOptions.use_ancestor_publication', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tryToFindInnerReadonlyDocWhenOpenUrl', full_name='com.gftchina.common.persistence.workspace.WorkspaceDataOptions.tryToFindInnerReadonlyDocWhenOpenUrl', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='shortcuts', full_name='com.gftchina.common.persistence.workspace.WorkspaceDataOptions.shortcuts', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='current_functional_block', full_name='com.gftchina.common.persistence.workspace.WorkspaceDataOptions.current_functional_block', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dependent_node_gids', full_name='com.gftchina.common.persistence.workspace.WorkspaceDataOptions.dependent_node_gids', index=6,
      number=7, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='active_publication', full_name='com.gftchina.common.persistence.workspace.WorkspaceDataOptions.active_publication', index=7,
      number=8, type=9, cpp_type=9, label=1,
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
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1685,
  serialized_end=2036,
)


_WORKSPACEINFO = _descriptor.Descriptor(
  name='WorkspaceInfo',
  full_name='com.gftchina.common.persistence.workspace.WorkspaceInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ui_data', full_name='com.gftchina.common.persistence.workspace.WorkspaceInfo.ui_data', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='io_edge_info', full_name='com.gftchina.common.persistence.workspace.WorkspaceInfo.io_edge_info', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ws_opt', full_name='com.gftchina.common.persistence.workspace.WorkspaceInfo.ws_opt', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ws_view_settings', full_name='com.gftchina.common.persistence.workspace.WorkspaceInfo.ws_view_settings', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2039,
  serialized_end=2363,
)


_SERVERINTERESTEDWSINFO = _descriptor.Descriptor(
  name='ServerInterestedWSInfo',
  full_name='com.gftchina.common.persistence.workspace.ServerInterestedWSInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ws_gid', full_name='com.gftchina.common.persistence.workspace.ServerInterestedWSInfo.ws_gid', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ws_name', full_name='com.gftchina.common.persistence.workspace.ServerInterestedWSInfo.ws_name', index=1,
      number=2, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='client_ws_info_data', full_name='com.gftchina.common.persistence.workspace.ServerInterestedWSInfo.client_ws_info_data', index=2,
      number=3, type=12, cpp_type=9, label=2,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='version_id', full_name='com.gftchina.common.persistence.workspace.ServerInterestedWSInfo.version_id', index=3,
      number=4, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dependent_nodes_json', full_name='com.gftchina.common.persistence.workspace.ServerInterestedWSInfo.dependent_nodes_json', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dependent_node_gids', full_name='com.gftchina.common.persistence.workspace.ServerInterestedWSInfo.dependent_node_gids', index=5,
      number=6, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='ws_properties', full_name='com.gftchina.common.persistence.workspace.ServerInterestedWSInfo.ws_properties', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
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
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2366,
  serialized_end=2600,
)

_VIEWSETTINGS.fields_by_name['view_mode'].enum_type = _VIEWSETTINGS_VIEWMODE
_VIEWSETTINGS.fields_by_name['mode_settings'].message_type = common_pb2._GFTSERIALIZABLE
_VIEWSETTINGS_VIEWMODE.containing_type = _VIEWSETTINGS
_IOEDGEINFO.fields_by_name['index'].enum_type = _IOEDGEINFO_OUTPUTINDEX
_IOEDGEINFO.fields_by_name['target_action'].enum_type = _IOEDGEINFO_TARGETACTIONS
_IOEDGEINFO_OUTPUTINDEX.containing_type = _IOEDGEINFO
_IOEDGEINFO_TARGETACTIONS.containing_type = _IOEDGEINFO
_UIDATA_COMMONSETTING.fields_by_name['disp_pattern'].enum_type = _UIDATA_COMMONSETTING_DISPLAYPATTERN
_UIDATA_COMMONSETTING.fields_by_name['tool_bar_button_names'].message_type = common_pb2._META
_UIDATA_COMMONSETTING.fields_by_name['available_new_node_type'].message_type = _NEWNODEINFO
_UIDATA_COMMONSETTING.containing_type = _UIDATA
_UIDATA_COMMONSETTING_DISPLAYPATTERN.containing_type = _UIDATA_COMMONSETTING
_UIDATA.fields_by_name['ui_type'].enum_type = uitypes_pb2._UITYPE
_UIDATA.fields_by_name['view_settings'].message_type = _VIEWSETTINGS
_UIDATA.fields_by_name['serialize'].message_type = common_pb2._GFTSERIALIZABLE
_UIDATA.fields_by_name['common_setting'].message_type = _UIDATA_COMMONSETTING
_WORKSPACEDATAOPTIONS.fields_by_name['layouts'].message_type = common_pb2._GFTSERIALIZABLE
_WORKSPACEDATAOPTIONS.fields_by_name['shortcuts'].message_type = common_pb2._META
_WORKSPACEINFO.fields_by_name['ui_data'].message_type = _UIDATA
_WORKSPACEINFO.fields_by_name['io_edge_info'].message_type = _IOEDGEINFO
_WORKSPACEINFO.fields_by_name['ws_opt'].message_type = _WORKSPACEDATAOPTIONS
_WORKSPACEINFO.fields_by_name['ws_view_settings'].message_type = _VIEWSETTINGS
_SERVERINTERESTEDWSINFO.fields_by_name['ws_properties'].message_type = common_pb2._META
DESCRIPTOR.message_types_by_name['ViewSettings'] = _VIEWSETTINGS
DESCRIPTOR.message_types_by_name['IOEdgeInfo'] = _IOEDGEINFO
DESCRIPTOR.message_types_by_name['NewNodeInfo'] = _NEWNODEINFO
DESCRIPTOR.message_types_by_name['UIData'] = _UIDATA
DESCRIPTOR.message_types_by_name['WorkspaceDataOptions'] = _WORKSPACEDATAOPTIONS
DESCRIPTOR.message_types_by_name['WorkspaceInfo'] = _WORKSPACEINFO
DESCRIPTOR.message_types_by_name['ServerInterestedWSInfo'] = _SERVERINTERESTEDWSINFO

ViewSettings = _reflection.GeneratedProtocolMessageType('ViewSettings', (_message.Message,), dict(
  DESCRIPTOR = _VIEWSETTINGS,
  __module__ = 'workspace_pb2'
  # @@protoc_insertion_point(class_scope:com.gftchina.common.persistence.workspace.ViewSettings)
  ))
_sym_db.RegisterMessage(ViewSettings)

IOEdgeInfo = _reflection.GeneratedProtocolMessageType('IOEdgeInfo', (_message.Message,), dict(
  DESCRIPTOR = _IOEDGEINFO,
  __module__ = 'workspace_pb2'
  # @@protoc_insertion_point(class_scope:com.gftchina.common.persistence.workspace.IOEdgeInfo)
  ))
_sym_db.RegisterMessage(IOEdgeInfo)

NewNodeInfo = _reflection.GeneratedProtocolMessageType('NewNodeInfo', (_message.Message,), dict(
  DESCRIPTOR = _NEWNODEINFO,
  __module__ = 'workspace_pb2'
  # @@protoc_insertion_point(class_scope:com.gftchina.common.persistence.workspace.NewNodeInfo)
  ))
_sym_db.RegisterMessage(NewNodeInfo)

UIData = _reflection.GeneratedProtocolMessageType('UIData', (_message.Message,), dict(

  CommonSetting = _reflection.GeneratedProtocolMessageType('CommonSetting', (_message.Message,), dict(
    DESCRIPTOR = _UIDATA_COMMONSETTING,
    __module__ = 'workspace_pb2'
    # @@protoc_insertion_point(class_scope:com.gftchina.common.persistence.workspace.UIData.CommonSetting)
    ))
  ,
  DESCRIPTOR = _UIDATA,
  __module__ = 'workspace_pb2'
  # @@protoc_insertion_point(class_scope:com.gftchina.common.persistence.workspace.UIData)
  ))
_sym_db.RegisterMessage(UIData)
_sym_db.RegisterMessage(UIData.CommonSetting)

WorkspaceDataOptions = _reflection.GeneratedProtocolMessageType('WorkspaceDataOptions', (_message.Message,), dict(
  DESCRIPTOR = _WORKSPACEDATAOPTIONS,
  __module__ = 'workspace_pb2'
  # @@protoc_insertion_point(class_scope:com.gftchina.common.persistence.workspace.WorkspaceDataOptions)
  ))
_sym_db.RegisterMessage(WorkspaceDataOptions)

WorkspaceInfo = _reflection.GeneratedProtocolMessageType('WorkspaceInfo', (_message.Message,), dict(
  DESCRIPTOR = _WORKSPACEINFO,
  __module__ = 'workspace_pb2'
  # @@protoc_insertion_point(class_scope:com.gftchina.common.persistence.workspace.WorkspaceInfo)
  ))
_sym_db.RegisterMessage(WorkspaceInfo)

ServerInterestedWSInfo = _reflection.GeneratedProtocolMessageType('ServerInterestedWSInfo', (_message.Message,), dict(
  DESCRIPTOR = _SERVERINTERESTEDWSINFO,
  __module__ = 'workspace_pb2'
  # @@protoc_insertion_point(class_scope:com.gftchina.common.persistence.workspace.ServerInterestedWSInfo)
  ))
_sym_db.RegisterMessage(ServerInterestedWSInfo)


# @@protoc_insertion_point(module_scope)