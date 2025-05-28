import 'package:flutter/material.dart';

class AppButtonTheme {
  AppButtonTheme._();

  static final lightButtonTheme = FilledButtonThemeData(
    style: ButtonStyle(
      backgroundColor: WidgetStatePropertyAll(Color(0xFF1F4BEA)),
      foregroundColor: WidgetStatePropertyAll(Colors.white),
      overlayColor: WidgetStatePropertyAll(Colors.black),
      shape: WidgetStatePropertyAll(RoundedRectangleBorder(borderRadius: BorderRadius.circular(20))),
      //side: WidgetStatePropertyAll(BorderSide(color: Colors.black))
    ),
  );
  static final darkButtonTheme = FilledButtonThemeData(
    style: ButtonStyle(
      backgroundColor: WidgetStatePropertyAll(Color(0xFF1F4BEA)),
      foregroundColor: WidgetStatePropertyAll(Colors.white),
      overlayColor: WidgetStatePropertyAll(Colors.white),
      shape: WidgetStatePropertyAll(RoundedRectangleBorder(borderRadius: BorderRadius.circular(20))),
      //side: WidgetStatePropertyAll(BorderSide(color: Colors.white))
    ),
  );
}
