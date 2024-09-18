from flask import Flask

ungdung = Flask(__name__)

@ungdung.route("/")

def text():
	return '''
	<html>
		<head>
		<title>Minh</title>
		</head>
		<body>
			<img src="static/images/1.png" alt="" width="300">

		</body>
	</html>
	'''

if __name__ == "__main__":
	ungdung.run(port=5000)