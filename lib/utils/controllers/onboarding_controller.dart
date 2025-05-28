import 'package:turd/pages/home_page.dart';
import 'package:flutter/widgets.dart';
import 'package:get/get.dart';
import 'package:turd/pages/main_screen.dart';

class OnboardingController extends GetxController{
  static OnboardingController get instance => Get.find();

  //variable
  final pageController = PageController();
  Rx<int> currentPageIndex = 0.obs;

  //update current index when page scroll
  void updatePageIndicator(index) => currentPageIndex.value = index;

  //jump to specific dot selected page
  void dotNavigationClick(index){
    currentPageIndex.value = index;
    pageController.jumpToPage(index);
  }

  //update current index and move to  naext page
  void nextPage(){
    if (currentPageIndex.value==2){
      Get.to(() => MainScreen());
    } else{
      int page = currentPageIndex.value + 1;
      pageController.jumpToPage(page);
    }
  }

  //update current index and jump to last page
  void skipPage(){
    currentPageIndex.value = 2;
    pageController.jumpToPage(2);
  }
}