import 'package:flutter/material.dart';
import 'app.dart';
import 'package:flutter_native_splash/flutter_native_splash.dart';
// import 'package:firebase_core/firebase_core.dart';

Future<void> main() async{
  WidgetsBinding widgetsBinding = WidgetsFlutterBinding.ensureInitialized();
  FlutterNativeSplash.preserve(widgetsBinding: widgetsBinding);
  runApp(const MyApp());
}
