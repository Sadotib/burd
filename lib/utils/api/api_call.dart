import 'package:http/http.dart' as http;
import 'dart:io';
import 'dart:convert';
import 'package:turd/utils/helper/bird_result.dart';

Future<void> predictBird(File imageFile) async {
  print("hello1");
  if (!await imageFile.exists()) {
    throw Exception('Image file does not exist at the given path');
  }

  var uri = Uri.parse('https://burd-production.up.railway.app/predict');
  var request = http.MultipartRequest('POST', uri);
  request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));
  print("hello2");
  var response = await request.send();
  print("hello3");
  if (response.statusCode == 200) {
    final resStr = await response.stream.bytesToString();
    final data = jsonDecode(resStr);
    String prediction = data['prediction'];
    double confidence = data['confidence'];
    print("Prediction: $prediction");
    print("Confidence: $confidence%");

    BirdResult().prediction = prediction;
    BirdResult().confidence = confidence;
  } else {
    throw Exception('Prediction failed with status: ${response.statusCode}');
  }
}
