from setuptools import setup, find_packages

setup(
    name='AURA-agent',
    version='0.1.0',
    packages=find_packages(), 
    
    install_requires=[
        'Flask',
        'python-dotenv',
        'google-genai',
        'cryptography',
    ],
    # to change my agent serivce start service endpoint
    entry_points={
        'console_scripts': [
            'aura-agent-start = my_agent.agent_service:start_server', 
        ],
    },
    author='Harpreet Singh',
    description='AURA - Autonomous Unified Resource Agent',
    license='unliscensed',
)