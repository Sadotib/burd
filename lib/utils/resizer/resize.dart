// import 'dart:io';
// import 'dart:typed_data';
// import 'package:image/image.dart' as img;
// import 'package:path_provider/path_provider.dart';

// Future<String> resizeImageTo224x224(String imagePath) async {
//   // Load the image from file
//   final imageBytes = await File(imagePath).readAsBytes();
//   final originalImage = img.decodeImage(imageBytes);

//   if (originalImage == null) {
//     throw Exception("Failed to decode image.");
//   }

//   // Resize the image to 224x224
//   final resizedImage = img.copyResize(originalImage, width: 224, height: 224);

//   // Save resized image to a temporary file
//   final tempDir = await getTemporaryDirectory();
//   final resizedImagePath = "${tempDir.path}/resized_image.png";

//   final resizedImageFile = File(resizedImagePath)
//     ..writeAsBytesSync(img.encodeJpg(resizedImage, quality: 100));

//   return resizedImageFile.path;
// }

// import 'dart:io';
// import 'package:flutter/material.dart';
// import 'package:image/image.dart' as img;
// import 'package:path_provider/path_provider.dart';

// Future<String> resizeImageTo224x224(String imagePath) async {
//   final imageBytes = await File(imagePath).readAsBytes();
//   final originalImage = img.decodeImage(imageBytes);

//   if (originalImage == null) {
//     throw Exception("Failed to decode image.");
//   }

//   final resizedImage = img.copyResize(originalImage, width: 224, height: 224);

  

// if (originalImage == null) throw Exception("Failed to decode image.");

//   // Ensure it's RGB (no alpha) and resize
//   // final rgbImage = img.bakeOrientation(originalImage); // handles EXIF rotation
//   // final resizedImage = img.copyResize(rgbImage, width: 224, height: 224);

//   // Save as JPG for consistency with TFLite model expectations
  
  
//   final tempDir = await getTemporaryDirectory();
//   final resizedImagePath = "${tempDir.path}/resized_image.jpg";
  

//   final resizedImageFile = File(resizedImagePath)
//     ..writeAsBytesSync(img.encodeJpg(resizedImage, quality: 100));
  
//   return resizedImageFile.path;
// }


import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;

Future<Float32List> imageToInputTensor(String imagePath) async {
  final imageBytes = await File(imagePath).readAsBytes();
  final decoded = img.decodeImage(imageBytes);

  if (decoded == null) {
    throw Exception("Image decode failed");
  }

  final resized = img.copyResize(decoded, width: 224, height: 224);
  final Float32List input = Float32List(1 * 224 * 224 * 3);
  int index = 0;

  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      final pixel = resized.getPixel(x, y);
      input[index++] = (img.getRed(pixel) / 127.5) - 1.0;
      input[index++] = (img.getGreen(pixel) / 127.5) - 1.0;
      input[index++] = (img.getBlue(pixel) / 127.5) - 1.0;
    }
  }

  return input;
}