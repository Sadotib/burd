import 'package:flutter/material.dart';
import 'package:get/get.dart';

class HelperFunction{
  

  static double getAppBarHeight(){
    return kToolbarHeight;
  }

  static double getBottomNavigationBarHeight(){
    return kBottomNavigationBarHeight;
  }
  
  static bool isDark(BuildContext context){
    bool isDark = Theme.of(context).brightness==Brightness.dark;
    
    return isDark;
  }
  //static bool isDark() => Get.theme.brightness == Brightness.dark;

  static double deviceHeight() {
      return MediaQuery.of(Get.context!).size.height;
  }
  // static double deviceWidth(){
  //   return Get.width;
  // }
   static double deviceWidth() {
      return MediaQuery.of(Get.context!).size.width;
   }
}
