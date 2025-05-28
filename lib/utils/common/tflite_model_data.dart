// lib/utils/tflite_model_data.dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:io';

late Interpreter interpreter;
late List<int> inputShape;
late List<int> outputShape;
late TensorType outputType;
late File imageFile;