import 'package:flutter/material.dart';

class AppIconButtonTheme{
  AppIconButtonTheme._();

  static final lightIconButtonTheme = IconButtonThemeData(
    style: ButtonStyle(
      iconColor: WidgetStatePropertyAll(Colors.white),
    )
  );
  static final darkIconButtonTheme = IconButtonThemeData(
    style: ButtonStyle(
      iconColor: WidgetStatePropertyAll(Colors.black),
    )
  );
  
}