import 'package:flutter/material.dart';
import 'package:get/state_manager.dart';
import 'package:turd/pages/home_page.dart';
import 'package:get/get.dart';

class NavigationMenu extends StatelessWidget {
  const NavigationMenu({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = Get.put(NavigationController());

    return Scaffold(
      bottomNavigationBar: Obx(
        () => NavigationBar(      //passing a child pointing to a function/widget
          backgroundColor:Colors.white,
          indicatorColor: Colors.white,
          indicatorShape: CircleBorder(),
          height: MediaQuery.of(context).size.height*0.07,
          selectedIndex: controller.selectedIndex.value,
          onDestinationSelected: (index) => controller.selectedIndex.value = index,
        
          destinations: [
            NavigationDestination(icon: Icon(Icons.home_filled), label: "Home"),
            NavigationDestination(icon: Icon(Icons.settings), label: "Settings"),
          ],
        ),
      ),
      body: Obx(() => controller.screens[controller.selectedIndex.value]),
    );
  }
}

class NavigationController extends GetxController{
  final Rx<int> selectedIndex = 0.obs;
  final screens = [HomePage(),Container()];
}

