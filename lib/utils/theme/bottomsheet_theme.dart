import 'package:flutter/material.dart';

class BottomSheetTheme{
  BottomSheetTheme._();

  static BottomSheetThemeData lightBottomSheetTheme = BottomSheetThemeData(
    
    dragHandleColor: Colors.white,
    modalBackgroundColor: Color(0xFF000000),
    showDragHandle: true,
  
  );

  static BottomSheetThemeData darkBottomSheetTheme = BottomSheetThemeData(
    
    dragHandleColor: Colors.black,
    modalBackgroundColor: Color(0xFFFFFFFF),
    showDragHandle: true,
  );
}