import 'package:flutter/material.dart';
import 'package:turd/utils/theme/button_theme.dart';
import 'package:turd/utils/theme/bottomsheet_theme.dart';
import 'package:turd/utils/theme/text_theme.dart';
import 'package:turd/utils/theme/appbar_theme.dart';
import 'package:turd/utils/theme/bottombar_theme.dart';
import 'package:turd/utils/theme/card_theme.dart';
import 'package:turd/utils/theme/iconbutton_theme.dart';
import 'package:turd/utils/theme/outlinebutton_theme.dart';

class AppTheme {
  AppTheme._();

  static ThemeData lightTheme = ThemeData(
    brightness: Brightness.light,
    scaffoldBackgroundColor: Colors.white,
    appBarTheme: TopBarTheme.lightAppBarTheme,

    filledButtonTheme: AppButtonTheme.lightButtonTheme,
    textTheme: AppTextTheme.lightTextTheme,
    bottomSheetTheme: BottomSheetTheme.lightBottomSheetTheme,
    bottomAppBarTheme: BottomBarTheme.lightBottomBarTheme,
    cardTheme: AppCardTheme.lightCardTheme,
    iconButtonTheme: AppIconButtonTheme.lightIconButtonTheme,
    outlinedButtonTheme: AppOutlinedButtonTheme.lightOutlinedButtonTheme,
  );
  static ThemeData darkTheme = ThemeData(
    brightness: Brightness.dark,
    scaffoldBackgroundColor: Colors.black,
    appBarTheme: TopBarTheme.darkAppBarTheme,
    filledButtonTheme: AppButtonTheme.darkButtonTheme,
    textTheme: AppTextTheme.darkTextTheme,
    bottomSheetTheme: BottomSheetTheme.darkBottomSheetTheme,
    bottomAppBarTheme: BottomBarTheme.darkBottomBarTheme,
    cardTheme: AppCardTheme.darkCardTheme,
    iconButtonTheme: AppIconButtonTheme.darkIconButtonTheme,
    outlinedButtonTheme: AppOutlinedButtonTheme.darkOutlinedButtonTheme,
  );
}

// ThemeData lightTheme = ThemeData(
//   brightness: Brightness.light,
//   scaffoldBackgroundColor: Colors.white,
//   appBarTheme: AppBarTheme(backgroundColor: Colors.white),
//   filledButtonTheme: AppButtonTheme.lightButtonTheme,
//   textTheme: AppTextTheme.lightTextTheme,
//   bottomSheetTheme: BottomSheetTheme.lightSheetTheme,
// );

// ThemeData darkTheme = ThemeData(
//   brightness: Brightness.dark,
//   scaffoldBackgroundColor: Colors.black,
//   appBarTheme: AppBarTheme(backgroundColor: Colors.black),
//   filledButtonTheme: AppButtonTheme.darkButtonTheme,
//   textTheme: AppTextTheme.darkTextTheme,
//   bottomSheetTheme: BottomSheetTheme.darkSheetTheme,
// );
