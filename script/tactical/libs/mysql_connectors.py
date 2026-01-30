import os
from os.path import join as jn
import contextlib
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader
from sshtunnel import SSHTunnelForwarder, open_tunnel
from sqlalchemy import create_engine


def ssh_tunnel_connection(ssh_parameters,hostname,port=22,allow_agent=False,debug_level=0):
	if 'ssh_password' in ssh_parameters.keys():
		ssh_tunnel = open_tunnel(ssh_parameters.get('ssh_hostname'),
										ssh_username=ssh_parameters.get('ssh_username'),
										ssh_password=ssh_parameters.get('ssh_password'),
										allow_agent=allow_agent,
										remote_bind_address=(hostname, port),
										debug_level=debug_level,
										set_keepalive=20.
										)
	else:
		ssh_tunnel = open_tunnel(ssh_parameters.get('ssh_hostname'),
										ssh_username=ssh_parameters.get('ssh_username'),
										ssh_pkey=ssh_parameters.get('ssh_pkey'),
										ssh_private_key_password=ssh_parameters.get('ssh_key_password',''),
										allow_agent=allow_agent,
										remote_bind_address=(hostname, port),
										debug_level=debug_level,
										set_keepalive=20.
										)
	ssh_tunnel.start()

	return ssh_tunnel

@contextmanager
def mysql_server(engine=None, hostname=None, port=None, username=None, password=None, database=None,
	connector='mysqldb', ssh_parameters=None, allow_agent=False, ssh_tunnel=None,
	debug_level = 0):
	"""
	If engine is given, yield simply engine.
	"""

	kill_engine = False
	if engine is None:
		kill_engine = True
		if ssh_tunnel is None and ssh_parameters is None:
			engine = create_engine('mysql+' + connector + '://' + username + ':' + password + '@' + hostname + '/' + database)
		else:
			if ssh_tunnel is None:
				ssh_tunnel = ssh_tunnel_connection(ssh_parameters,hostname,port,allow_agent,debug_level)
				engine = create_engine('mysql+' + connector + '://' + username + ':' + password + '@127.0.0.1:%s/' % ssh_tunnel.local_bind_port + database)

	try:
		yield {'engine':engine, 'ssh_tunnel':ssh_tunnel}
	finally:
		if kill_engine:
			engine.dispose()
			if not ssh_tunnel is None:
				ssh_tunnel.stop()

@contextmanager
def mysql_connection(profile=None, path_profile=None, **kwargs):
	"""
	profile can be any string corresponding to a file 'db_profile_credentials.py'

	profile is ignored if a non-None engine is passed in argument.
	"""
	print(profile)
	if profile is None and (not 'engine' in kwargs.keys() or kwargs['engine'] is None):
		yield {'engine':None, 'ssh_tunnel':None}
	else:
		if not 'engine' in kwargs.keys() or kwargs['engine'] is None:
			name = profile + '_credentials'
			if path_profile	is None:
				import libs.mysql_connectors as mysql_connectors
				path_profile = os.path.dirname(os.path.dirname(os.path.abspath(mysql_connectors.__file__)))

			print(path_profile)

			cred = SourceFileLoader(name, jn(path_profile, name + '.py')).load_module()

			for par in ['hostname', 'username', 'password']:
				kwargs[par] = kwargs.get(par, cred.__getattribute__(par))

			if kwargs.get('database') is None:
				kwargs['database'] = cred.__getattribute__('database')

			kwargs['port'] = kwargs.get('port', 3306)
			try:
				kwargs['connector']=cred.__getattribute__('mysql_connector')
			except:
				kwargs['connector'] = 'mysqldb'

			print ('DB connection to', kwargs['hostname'], end=" ")

			if not 'ssh_tunnel' in kwargs.keys() or kwargs['ssh_tunnel'] is None:
				if 'ssh_username' in cred.__dir__():
					ssh_auth = cred.__getattribute__('ssh_auth')
					kwargs['ssh_parameters'] = kwargs.get('ssh_parameters', {})
					if ssh_auth=='password':
						for par in ['ssh_username', 'ssh_password', 'ssh_hostname']:
							kwargs['ssh_parameters'][par] = kwargs['ssh_parameters'].get(par, cred.__getattribute__(par))	
					elif ssh_auth=='key':
						for par in ['ssh_username', 'ssh_key_password', 'ssh_pkey', 'ssh_hostname']:
							kwargs['ssh_parameters'][par] = kwargs['ssh_parameters'].get(par, cred.__getattribute__(par))	
						kwargs['allow_agent'] = True

					print ('with ssh tunneling through', kwargs['ssh_parameters']['ssh_hostname'])
				else:
					print ()
		with mysql_server(**kwargs) as connection:
			yield connection