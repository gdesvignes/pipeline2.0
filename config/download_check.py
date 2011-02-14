import config_types

download = config_types.ConfigList('download')
download.add_config('api_service_url', config_types.StrConfig())
download.add_config('api_username', config_types.StrConfig())
download.add_config('api_password', config_types.StrConfig())
download.add_config('temp', config_types.DirConfig())
download.add_config('space_to_use', config_types.IntConfig())
download.add_config('numdownloads', config_types.IntConfig())
download.add_config('numrestores', config_types.IntConfig())
download.add_config('numretries', config_types.IntConfig())
